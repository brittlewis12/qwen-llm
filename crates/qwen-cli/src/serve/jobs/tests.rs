use super::state::*;
use super::store::FaultPoint;
use super::store::{JobStore, Limits, Start, StoreError};
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct TestRoot(pub(super) PathBuf);
impl TestRoot {
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "qwen-jobs-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        Self(path)
    }
    pub(super) fn open(&self) -> JobStore {
        JobStore::open(&self.0, Limits::default()).unwrap()
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn request() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/lens_http_v1/request.json"
    ))
    .unwrap()
}
fn accepted(store: &JobStore, key: &str, observations: bool) -> String {
    let accepted = store.accept(key, &request(), observations).unwrap();
    assert!(accepted.created);
    accepted.status.id
}
fn counters() -> Counters {
    Counters {
        prompt_tokens: 2,
        consumed_prompt_tokens: 2,
        sampled_tokens: 2,
        consumed_generated_tokens: 1,
    }
}
fn completed(store: &JobStore, id: &str) {
    store
        .finish_generation(id, StopReason::TokenLimit, counters(), None)
        .unwrap();
    store.finalize(id, None).unwrap();
}
fn readout() -> Value {
    json!({"kind":"readout", "readout_id":"r1", "scores":[]})
}
fn failure() -> JobError {
    JobError {
        r#type: "server_error".into(),
        code: "observation_failed".into(),
        param: None,
        message: "writer failed".into(),
    }
}

#[test]
fn encoded_batches_match_value_records_and_publish_progress_once() {
    let root = TestRoot::new();
    let store = root.open();
    let a = accepted(&store, "values", true);
    let b = accepted(&store, "encoded", true);
    store.start(&a, 2).unwrap();
    store.start(&b, 2).unwrap();
    let records = [
        json!({"kind":"readout","scores":[1.25,-0.0],"text":"line\nquoted\"","token":u64::MAX}),
        readout(),
    ];
    let bytes = records
        .iter()
        .map(|r| serde_json::to_vec(r).unwrap())
        .collect::<Vec<_>>();
    let progress = Counters {
        prompt_tokens: 2,
        consumed_prompt_tokens: 1,
        ..Default::default()
    };
    let revision = store.status(&b).unwrap().revision;
    store.append(&a, &records).unwrap();
    store
        .progress(&a, Phase::Prefill, progress.clone())
        .unwrap();
    let status = store
        .append_encoded_progress(
            &b,
            &bytes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            Phase::Prefill,
            progress.clone(),
        )
        .unwrap();
    assert_eq!(status.revision, revision + 1);
    assert_eq!(status.generation.counters, progress);
    assert_eq!(status.observations.committed_records, 2);
    assert_eq!(
        fs::read(root.0.join(&a).join("records.jsonl")).unwrap(),
        fs::read(root.0.join(&b).join("records.jsonl")).unwrap()
    );
    assert_eq!(
        store.result(&a, None, 10).unwrap().records,
        store.result(&b, None, 10).unwrap().records
    );
    let c = accepted(&store, "progress-only", false);
    store.start(&c, 2).unwrap();
    let status = store
        .append_encoded_progress(&c, &[], Phase::Prefill, progress)
        .unwrap();
    assert_eq!(status.generation.counters.consumed_prompt_tokens, 1);
    assert!(!status.result.available);
    assert!(store.result(&c, None, 10).unwrap().records.is_empty());
}

#[test]
fn encoded_batch_validation_refuses_framing_and_owned_fields_without_publication() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "invalid-encoded", true);
    store.start(&id, 2).unwrap();
    let status = store.status(&id).unwrap();
    for bytes in [
        b"[]".as_slice(),
        br#"{}"#,
        br#"{"kind":"readout","seq":null}"#,
        br#"{"kind":"readout","seq":1}"#,
        br#"{"kind":"retained_array"}"#,
        br#"{"kind":"a","kind":"b"}"#,
        br#"{"kind":"a"}{}"#,
        b"{\"kind\":\"a\",\n\"value\":1}",
        br#"{"kind":"a","bad":[}"#,
    ] {
        assert!(
            store
                .append_encoded_progress(
                    &id,
                    &[bytes],
                    Phase::Prefill,
                    Counters {
                        prompt_tokens: 2,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        assert_eq!(store.status(&id).unwrap(), status);
    }
    assert!(
        store
            .append_encoded_progress(
                &id,
                &[br#"{"kind":"ok"}"#],
                Phase::Decode,
                Counters {
                    prompt_tokens: 2,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(store.result(&id, None, 10).unwrap().records.is_empty());
}

#[test]
fn encoded_limits_include_sequence_digits_and_jsonl_newlines() {
    let root = TestRoot::new();
    let raw = br#"{"kind":"test"}"#;
    let bytes = raw.len() + 9;
    let store = JobStore::open(
        &root.0,
        Limits {
            max_record_bytes: bytes,
            max_batch_bytes: bytes,
            ..Default::default()
        },
    )
    .unwrap();
    let id = accepted(&store, "encoded-limits", false);
    store.start(&id, 2).unwrap();
    let progress = Counters {
        prompt_tokens: 2,
        ..Default::default()
    };
    assert!(
        store
            .append_encoded_progress(&id, &[raw, raw], Phase::Prefill, progress.clone())
            .is_err()
    );
    for _ in 0..10 {
        store
            .append_encoded_progress(&id, &[raw], Phase::Prefill, progress.clone())
            .unwrap();
    }
    assert!(
        store
            .append_encoded_progress(&id, &[raw], Phase::Prefill, progress)
            .is_err()
    );
    assert_eq!(
        fs::metadata(root.0.join(id).join("records.jsonl"))
            .unwrap()
            .len(),
        (10 * bytes) as u64
    );
}

#[test]
fn encoded_batch_faults_keep_records_and_progress_on_the_same_watermark() {
    for point in [
        FaultPoint::RecordPartialWrite,
        FaultPoint::RecordSync,
        FaultPoint::SnapshotRenamed,
    ] {
        let root = TestRoot::new();
        let store = root.open();
        let id = accepted(&store, "encoded-fault", true);
        store.start(&id, 2).unwrap();
        store.fail_once(point);
        let records = [
            br#"{"kind":"readout"}"#.as_slice(),
            br#"{"kind":"residual_pair"}"#,
        ];
        assert!(
            store
                .append_encoded_progress(
                    &id,
                    &records,
                    Phase::Prefill,
                    Counters {
                        prompt_tokens: 2,
                        consumed_prompt_tokens: 1,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        assert!(store.result(&id, None, 10).unwrap().records.is_empty());
        drop(store);
        let store = root.open();
        assert_eq!(
            store.result(&id, None, 10).unwrap().records.len(),
            if point == FaultPoint::SnapshotRenamed {
                2
            } else {
                0
            }
        );
        assert_eq!(
            store
                .status(&id)
                .unwrap()
                .generation
                .counters
                .consumed_prompt_tokens,
            u64::from(point == FaultPoint::SnapshotRenamed)
        );
    }
}

#[test]
fn acceptance_survives_restart_and_never_requeues() {
    let root = TestRoot::new();
    let id;
    {
        let store = root.open();
        id = accepted(&store, "same-key", true);
        assert_eq!(store.status(&id).unwrap().state, JobState::Queued);
        assert_eq!(store.request(&id).unwrap(), request());
    }
    let store = root.open();
    let status = store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Interrupted);
    assert_eq!(
        status.generation.stop_reason,
        Some(StopReason::ServerRestart)
    );
    assert!(status.result.complete);
    assert_eq!(store.start(&id, 2).unwrap(), Start::Settled);
    let retry = store.accept("same-key", &request(), true).unwrap();
    assert!(!retry.created);
    assert_eq!(retry.status, status);
    assert_eq!(store.history(None, 10).unwrap().jobs.len(), 1);
}

#[test]
fn concurrent_idempotent_acceptance_produces_one_job() {
    let root = TestRoot::new();
    let store = root.open();
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let store = &store;
                scope.spawn(move || store.accept("same", &request(), true).unwrap())
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|result| result.created).count(), 1);
    assert!(
        results
            .iter()
            .all(|result| result.status.id == results[0].status.id)
    );
    let mut changed = request();
    changed["generation"]["max_new_tokens"] = json!(99);
    assert!(matches!(
        store.accept("same", &changed, true),
        Err(StoreError::Conflict)
    ));
}

#[test]
fn active_capacity_does_not_reject_idempotent_lookup_or_terminal_history() {
    let root = TestRoot::new();
    let store = JobStore::open(
        &root.0,
        Limits {
            max_active_jobs: 1,
            ..Limits::default()
        },
    )
    .unwrap();
    let id = accepted(&store, "one", false);
    assert!(!store.accept("one", &request(), false).unwrap().created);
    assert!(matches!(
        store.accept("two", &request(), false),
        Err(StoreError::Full)
    ));
    store.cancel(&id).unwrap();
    accepted(&store, "two", false);
    assert_eq!(store.history(None, 10).unwrap().jobs.len(), 2);
}

#[test]
fn queued_cancel_is_durable_and_cannot_be_dispatched() {
    let root = TestRoot::new();
    let id;
    let cancelled;
    {
        let store = root.open();
        id = accepted(&store, "cancel", true);
        cancelled = store.cancel(&id).unwrap();
        assert_eq!(cancelled.state, JobState::Cancelled);
        assert!(cancelled.cancel_requested);
        assert!(store.control(&id).unwrap().checkpoint().is_err());
        assert_eq!(store.start(&id, 2).unwrap(), Start::Settled);
        assert_eq!(store.cancel(&id).unwrap(), cancelled);
    }
    assert_eq!(root.open().status(&id).unwrap(), cancelled);
}

#[test]
fn running_cancel_is_a_request_not_a_claim_of_gpu_completion() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "cancel", true);
    store.start(&id, 2).unwrap();
    let status = store.cancel(&id).unwrap();
    assert_eq!(status.state, JobState::Running);
    assert_eq!(status.generation.state, GenerationState::Running);
    assert!(!status.result.complete);
    let control = store.control(&id).unwrap();
    assert!(control.checkpoint().is_err());
    store
        .finish_generation(
            &id,
            StopReason::Cancelled,
            Counters {
                prompt_tokens: 2,
                ..Counters::default()
            },
            None,
        )
        .unwrap();
    let status = store.finalize(&id, None).unwrap();
    assert_eq!(status.state, JobState::Cancelled);
    assert_eq!(status.observations.state, ObservationState::Partial);
}

#[test]
fn dropping_history_readers_does_not_cancel_execution() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "disconnect", false);
    store.start(&id, 2).unwrap();
    drop(store.history(None, 10).unwrap());
    drop(store.status(&id).unwrap());
    drop(store.request(&id).unwrap());
    drop(store.result(&id, None, 1).unwrap());
    drop(store.control(&id).unwrap());
    assert!(store.control(&id).unwrap().checkpoint().is_ok());
}

#[test]
fn committed_pages_are_immutable_and_live_end_cursor_resumes() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "pages", true);
    store.start(&id, 2).unwrap();
    let empty = store.result(&id, None, 1).unwrap();
    assert!(empty.records.is_empty());
    assert!(!empty.complete);
    store
        .append(
            &id,
            &[
                json!({"kind":"prepared_input", "prompt_text":"exact\n"}),
                readout(),
            ],
        )
        .unwrap();
    let first = store.result(&id, empty.next_cursor.as_deref(), 1).unwrap();
    assert_eq!(first.records[0]["seq"], 0);
    let first_record = first.records[0].clone();
    let second = store.result(&id, first.next_cursor.as_deref(), 1).unwrap();
    assert_eq!(second.records[0]["seq"], 1);
    let waiting = store.result(&id, second.next_cursor.as_deref(), 1).unwrap();
    assert!(waiting.records.is_empty());
    assert_eq!(waiting.next_cursor, second.next_cursor);
    store
        .append(
            &id,
            &[json!({"kind":"sampled_token", "token_id":7, "consumed":false})],
        )
        .unwrap();
    completed(&store, &id);
    let third = store
        .result(&id, waiting.next_cursor.as_deref(), 1)
        .unwrap();
    assert!(third.complete);
    assert_eq!(third.records[0]["seq"], 2);
    assert_eq!(third.next_cursor, None);
    assert_eq!(store.result(&id, None, 1).unwrap().records[0], first_record);
    let status = store.status(&id).unwrap();
    assert_eq!(status.observations.committed_records, 1);
    assert_eq!(status.observations.state, ObservationState::Complete);
    assert!(store.append(&id, &[readout()]).is_err());
}

#[test]
fn observation_failure_does_not_rewrite_completed_generation() {
    let root = TestRoot::new();
    let id;
    let terminal;
    {
        let store = root.open();
        id = accepted(&store, "failed-observation", true);
        store.start(&id, 2).unwrap();
        store.append(&id, &[readout()]).unwrap();
        let generated = store
            .finish_generation(&id, StopReason::TokenLimit, counters(), None)
            .unwrap();
        assert_eq!(generated.state, JobState::Finalizing);
        assert!(!generated.result.complete);
        terminal = store.finalize(&id, Some(failure())).unwrap();
        assert_eq!(terminal.state, JobState::Failed);
        assert_eq!(terminal.generation, generated.generation);
        assert_eq!(terminal.observations.state, ObservationState::Partial);
    }
    assert_eq!(root.open().status(&id).unwrap(), terminal);
}

#[test]
fn restart_after_generation_preserves_outcome_and_committed_observations() {
    let root = TestRoot::new();
    let id;
    let generation;
    {
        let store = root.open();
        id = accepted(&store, "finalizing", true);
        store.start(&id, 2).unwrap();
        store.append(&id, &[readout()]).unwrap();
        generation = store
            .finish_generation(&id, StopReason::TokenLimit, counters(), None)
            .unwrap()
            .generation;
    }
    let store = root.open();
    let status = store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Interrupted);
    assert_eq!(status.generation, generation);
    assert_eq!(status.observations.state, ObservationState::Partial);
    let page = store.result(&id, None, 10).unwrap();
    assert_eq!(page.records.len(), 1);
    assert!(page.complete);
}

#[test]
fn recovery_discards_only_bytes_beyond_durable_watermark() {
    let root = TestRoot::new();
    let id;
    let committed;
    {
        let store = root.open();
        id = accepted(&store, "torn-tail", true);
        store.start(&id, 2).unwrap();
        store.append(&id, &[readout()]).unwrap();
        committed = fs::read(root.0.join(&id).join("records.jsonl")).unwrap();
    }
    let path = root.0.join(&id).join("records.jsonl");
    let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
    file.write_all(b"{\"seq\":1,\"kind\":\"readout\"}\n{torn")
        .unwrap();
    file.sync_all().unwrap();
    let store = root.open();
    assert_eq!(fs::read(&path).unwrap(), committed);
    assert_eq!(store.result(&id, None, 10).unwrap().records.len(), 1);
}

#[test]
fn missing_committed_bytes_fail_closed() {
    let root = TestRoot::new();
    let id;
    {
        let store = root.open();
        id = accepted(&store, "truncated", true);
        store.start(&id, 2).unwrap();
        store.append(&id, &[readout()]).unwrap();
    }
    fs::write(root.0.join(id).join("records.jsonl"), []).unwrap();
    assert!(JobStore::open(&root.0, Limits::default()).is_err());
}

#[test]
fn duplicate_owner_and_symlinked_root_are_rejected() {
    let root = TestRoot::new();
    let store = root.open();
    assert!(JobStore::open(&root.0, Limits::default()).is_err());
    let alias = TestRoot::new();
    std::os::unix::fs::symlink(&root.0, &alias.0).unwrap();
    assert!(JobStore::open(&alias.0, Limits::default()).is_err());
    fs::remove_file(&alias.0).unwrap();
    drop(store);
    root.open();
}

#[test]
fn failed_snapshot_publication_fences_job_without_changing_visible_records() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "failed-publication", true);
    store.start(&id, 2).unwrap();
    let status = store.status(&id).unwrap();
    fs::create_dir(root.0.join(&id).join(".status.next")).unwrap();
    assert!(store.append(&id, &[readout()]).is_err());
    assert!(store.status(&id).is_err());
    assert_eq!(store.history(None, 10).unwrap().jobs[0], status);
    assert!(store.result(&id, None, 10).unwrap().records.is_empty());
    assert!(store.control(&id).unwrap().checkpoint().is_err());
    assert!(store.append(&id, &[readout()]).is_err());
    fs::remove_dir(root.0.join(&id).join(".status.next")).unwrap();
    drop(store);
    let recovered = root.open();
    assert!(recovered.result(&id, None, 10).unwrap().records.is_empty());
    assert_eq!(recovered.status(&id).unwrap().state, JobState::Interrupted);
}

#[test]
fn invalid_progress_and_terminal_rewrites_are_rejected() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "progress", false);
    assert!(store.progress(&id, Phase::Decode, counters()).is_err());
    store.start(&id, 2).unwrap();
    assert!(store.start(&id, 2).is_err());
    assert!(
        store
            .progress(
                &id,
                Phase::Decode,
                Counters {
                    consumed_prompt_tokens: 1,
                    ..counters()
                }
            )
            .is_err()
    );
    store.progress(&id, Phase::Decode, counters()).unwrap();
    assert!(store.progress(&id, Phase::Prefill, counters()).is_err());
    assert!(
        store
            .progress(
                &id,
                Phase::Decode,
                Counters {
                    sampled_tokens: 1,
                    ..counters()
                }
            )
            .is_err()
    );
    assert!(store.finalize(&id, None).is_err());
    completed(&store, &id);
    let status = store.status(&id).unwrap();
    assert_eq!(store.cancel(&id).unwrap(), status);
    assert!(
        store
            .finish_generation(&id, StopReason::Cancelled, counters(), None)
            .is_err()
    );
}

#[test]
fn pages_enforce_byte_bounds_and_reject_cross_job_cursors() {
    let root = TestRoot::new();
    let limits = Limits {
        max_record_bytes: 512,
        max_page_bytes: 1536,
        ..Limits::default()
    };
    let store = JobStore::open(&root.0, limits).unwrap();
    let id = accepted(&store, "bytes", true);
    let other = accepted(&store, "other", false);
    store.start(&id, 2).unwrap();
    let record = json!({"kind":"readout", "text":"x".repeat(350)});
    store.append(&id, &[record.clone(), record]).unwrap();
    assert!(
        store
            .append(&id, &[json!({"kind":"readout", "text":"x".repeat(512)})])
            .is_err()
    );
    assert!(
        store
            .append(&id, &[json!({"kind":"readout", "seq":2})])
            .is_err()
    );
    let page = store.result(&id, None, 10).unwrap();
    assert_eq!(page.records.len(), 1);
    assert!(serde_json::to_vec(&page).unwrap().len() <= limits.max_page_bytes);
    assert!(
        store
            .result(&other, page.next_cursor.as_deref(), 10)
            .is_err()
    );
    assert!(store.result(&id, Some("../../request.json"), 10).is_err());
    assert!(store.result(&id, None, 0).is_err());
    assert!(store.history(Some("../../request.json"), 10).is_err());
}

#[test]
fn history_cursor_does_not_shift_when_new_jobs_arrive() {
    let root = TestRoot::new();
    let store = root.open();
    let first = accepted(&store, "first", false);
    let second = accepted(&store, "second", false);
    let page = store.history(None, 1).unwrap();
    assert_eq!(page.jobs[0].id, second);
    accepted(&store, "third", false);
    let older = store.history(page.next_cursor.as_deref(), 1).unwrap();
    assert_eq!(older.jobs[0].id, first);
    assert_eq!(older.next_cursor, None);
    assert_eq!(page.request_previews.len(), 1);
    assert!(page.request_previews.contains_key(&second));
    assert_eq!(older.request_previews.len(), 1);
    assert!(older.request_previews.contains_key(&first));
}

#[test]
fn history_previews_rebuild_without_migration_and_poll_without_request_io() {
    let root = TestRoot::new();
    let mut authored = request();
    authored["input"]["messages"] = json!([
        {"role":"user","content":"first"},
        {"role":"assistant","content":"not the preview"},
        {"role":"user","content":"last ".repeat(100)}
    ]);
    let (id, original);
    {
        let store = root.open();
        id = store.accept("preview", &authored, false).unwrap().status.id;
        original = store.history(None, 1).unwrap().request_previews;
        let preview = original[&id].as_ref().unwrap();
        assert_eq!(preview.message_index, 2);
        assert_eq!(preview.message_count, 3);
        assert!(preview.truncated);
        assert_eq!(preview.text, "last ".repeat(48));
        assert_eq!(store.request(&id).unwrap(), authored);
    }
    let store = root.open();
    let snapshot_path = root.0.join(&id).join("status.json");
    let before = fs::read(&snapshot_path).unwrap();
    let path = root.0.join(&id).join("request.json");
    let hidden = path.with_extension("held");
    fs::rename(&path, &hidden).unwrap();
    for _ in 0..3 {
        let page = store.history(None, 1).unwrap();
        assert_eq!(page.request_previews, original);
        assert_eq!(page.jobs[0].state, JobState::Interrupted);
        assert!(!page.jobs[0].cancel_requested);
    }
    fs::rename(hidden, path).unwrap();
    assert_eq!(fs::read(snapshot_path).unwrap(), before);
    assert_eq!(store.request(&id).unwrap(), authored);
    assert!(!store.accept("preview", &authored, false).unwrap().created);
    let unknown = store
        .accept("legacy", &json!({"legacy":"request"}), false)
        .unwrap()
        .status
        .id;
    assert_eq!(
        store.history(None, 1).unwrap().request_previews[&unknown],
        None
    );
}

#[test]
fn status_wire_shape_matches_the_frontend_fixture() {
    let fixture: JobStatus = serde_json::from_str(include_str!(
        "../../../tests/fixtures/lens_http_v1/status.json"
    ))
    .unwrap();
    assert_eq!(fixture.state, JobState::Running);
    assert_eq!(fixture.generation.counters.sampled_tokens, 9);
    assert_eq!(
        serde_json::to_value(fixture).unwrap(),
        serde_json::from_str::<Value>(include_str!(
            "../../../tests/fixtures/lens_http_v1/status.json"
        ))
        .unwrap()
    );
}

#[test]
fn retained_arrays_publish_with_records_and_survive_restart_without_an_executor() {
    let root = TestRoot::new();
    let bytes: Vec<u8> = [1.0f32, -2.5, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let id;
    {
        let store = root.open();
        id = store
            .accept_with_archive("arrays", &request(), true, 24)
            .unwrap()
            .status
            .id;
        store.start(&id, 2).unwrap();
        assert!(
            store
                .append(&id, &[json!({"kind":"retained_array"})])
                .is_err()
        );
        store
            .append_array(
                &id,
                json!({"key":"source-0-1","quantity":"source_residual"}),
                &bytes,
            )
            .unwrap();
        assert_eq!(store.array(&id, 0).unwrap(), bytes);
        assert!(store.array(&id, 1).is_err());
        assert!(store.array(&id, u64::MAX).is_err());
        assert!(
            store
                .append_array(&id, json!({"array":{}}), &bytes)
                .is_err()
        );
        let first = store.result(&id, None, 10).unwrap();
        assert_eq!(first.records[0]["array"]["length"], 3);
        assert_eq!(first.records[0]["array"]["byte_length"], 12);
        assert_eq!(
            first.records[0]["array"]["url"],
            format!("/v1/lens/jobs/{id}/arrays/0")
        );
        store.cancel(&id).unwrap();
        store
            .append_array(&id, json!({"key":"logits-0-1-plain"}), &bytes)
            .unwrap();
        assert!(
            store
                .append_array(&id, json!({"key":"over"}), &bytes)
                .is_err()
        );
        assert_eq!(store.status(&id).unwrap().observations.committed_records, 0);
    }
    let store = root.open();
    assert_eq!(store.array(&id, 0).unwrap(), bytes);
    assert_eq!(store.status(&id).unwrap().state, JobState::Interrupted);
    assert!(store.status(&id).unwrap().cancel_requested);
    assert_eq!(store.result(&id, None, 10).unwrap().records.len(), 2);
}

#[test]
fn array_failure_boundaries_recover_both_watermarks_without_exposing_tails() {
    for point in [
        FaultPoint::ArrayPartialWrite,
        FaultPoint::ArraySync,
        FaultPoint::RecordPartialWrite,
        FaultPoint::RecordSync,
        FaultPoint::SnapshotRenamed,
    ] {
        let root = TestRoot::new();
        let bytes = 2.0f32.to_le_bytes();
        let id;
        {
            let store = root.open();
            id = store
                .accept_with_archive("fault", &request(), true, 16)
                .unwrap()
                .status
                .id;
            store.start(&id, 2).unwrap();
            store.fail_once(point);
            assert!(
                store
                    .append_array(&id, json!({"key":"first"}), &bytes)
                    .is_err()
            );
            assert!(store.array(&id, 0).is_err());
            assert!(store.result(&id, None, 10).unwrap().records.is_empty());
        }
        let store = root.open();
        let committed = point == FaultPoint::SnapshotRenamed;
        assert_eq!(
            fs::metadata(root.0.join(&id).join("arrays.bin"))
                .unwrap()
                .len(),
            if committed { 4 } else { 0 }
        );
        assert_eq!(
            store.result(&id, None, 10).unwrap().records.len(),
            usize::from(committed)
        );
        if committed {
            assert_eq!(store.array(&id, 0).unwrap(), bytes);
        } else {
            assert!(store.array(&id, 0).is_err());
        }
    }
}

#[test]
fn corrupt_committed_arrays_fail_closed_at_read_and_recovery() {
    for damage in ["missing", "truncated", "mutated", "descriptor"] {
        let root = TestRoot::new();
        let store = root.open();
        let id = store
            .accept_with_archive("corrupt", &request(), true, 4)
            .unwrap()
            .status
            .id;
        store.start(&id, 2).unwrap();
        store
            .append_array(&id, json!({"key":"a"}), &1.0f32.to_le_bytes())
            .unwrap();
        let path = root.0.join(&id).join("arrays.bin");
        match damage {
            "missing" => fs::remove_file(path).unwrap(),
            "truncated" => fs::write(path, [0, 0]).unwrap(),
            "mutated" => fs::write(path, 2.0f32.to_le_bytes()).unwrap(),
            _ => {
                let path = root.0.join(&id).join("records.jsonl");
                let text = fs::read_to_string(&path).unwrap();
                fs::write(path, text.replace("\"offset\":0", "\"offset\":1")).unwrap();
            }
        }
        assert!(store.array(&id, 0).is_err());
        drop(store);
        assert!(JobStore::open(&root.0, Limits::default()).is_err());
    }
}

#[test]
fn queued_archive_reservations_count_before_execution_and_do_not_double_charge_payloads() {
    let root = TestRoot::new();
    let limits = Limits {
        max_store_bytes: 5 * 1024 * 1024,
        ..Limits::default()
    };
    let store = JobStore::open(&root.0, limits).unwrap();
    let id = store
        .accept_with_archive("one", &request(), true, 4 * 1024 * 1024)
        .unwrap()
        .status
        .id;
    assert!(matches!(
        store.accept_with_archive("two", &request(), true, 1024 * 1024),
        Err(StoreError::Full)
    ));
    assert!(
        !store
            .accept_with_archive("one", &request(), true, 4 * 1024 * 1024)
            .unwrap()
            .created
    );
    store.start(&id, 2).unwrap();
    store
        .append_array(&id, json!({"key":"large"}), &vec![0; 4 * 1024 * 1024])
        .unwrap();
}

#[test]
fn observation_policy_is_part_of_idempotency_identity() {
    let root = TestRoot::new();
    let store = root.open();
    accepted(&store, "policy", false);
    assert!(matches!(
        store.accept("policy", &request(), true),
        Err(StoreError::Conflict)
    ));
}

#[test]
fn request_object_order_is_irrelevant_but_operation_order_is_not() {
    let root = TestRoot::new();
    let store = root.open();
    let one: Value = serde_json::from_str(r#"{"b":{"x":1,"y":2},"a":[1,2]}"#).unwrap();
    let reordered: Value = serde_json::from_str(r#"{"a":[1,2],"b":{"y":2,"x":1}}"#).unwrap();
    assert!(store.accept("order", &one, false).unwrap().created);
    assert!(!store.accept("order", &reordered, false).unwrap().created);
    let changed = json!({"a":[2,1], "b":{"x":1,"y":2}});
    assert!(matches!(
        store.accept("order", &changed, false),
        Err(StoreError::Conflict)
    ));
    drop(store);
    assert!(
        !root
            .open()
            .accept("order", &reordered, false)
            .unwrap()
            .created
    );
}

#[test]
fn large_errors_are_rejected_before_mutating_recoverable_status() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "large-error", true);
    store.start(&id, 2).unwrap();
    let before = store.status(&id).unwrap();
    let error = JobError {
        message: "x".repeat(64 * 1024),
        ..failure()
    };
    assert!(matches!(
        store.finish_generation(&id, StopReason::ExecutionError, counters(), Some(error)),
        Err(StoreError::Invalid(_))
    ));
    assert_eq!(store.status(&id).unwrap(), before);
    drop(store);
    assert_eq!(
        root.open().status(&id).unwrap().state,
        JobState::Interrupted
    );
}

#[test]
fn baseline_artifact_failure_keeps_observations_not_requested() {
    for point in [FaultPoint::RecordPartialWrite, FaultPoint::RecordSync] {
        let root = TestRoot::new();
        let store = root.open();
        let id = accepted(&store, "baseline-artifact", false);
        store.start(&id, 2).unwrap();
        store
            .append(&id, &[json!({"kind":"prepared_input"})])
            .unwrap();
        let generated = store
            .finish_generation(&id, StopReason::TokenLimit, counters(), None)
            .unwrap();
        store.fail_once(point);
        assert!(
            store
                .append(&id, &[json!({"kind":"sampled_token", "token_id":7})])
                .is_err()
        );
        let finished = store.finalize(&id, Some(failure())).unwrap();
        assert_eq!(finished.generation, generated.generation);
        assert_eq!(finished.state, JobState::Failed);
        assert_eq!(finished.observations.state, ObservationState::NotRequested);
        assert_eq!(finished.observations.error, None);
        assert_eq!(finished.result.error, Some(failure()));
        assert_eq!(store.result(&id, None, 10).unwrap().records.len(), 1);
        drop(store);
        let store = root.open();
        assert_eq!(store.status(&id).unwrap(), finished);
        let page = store.result(&id, None, 10).unwrap();
        assert_eq!(page.records.len(), 1);
        assert!(page.complete);
    }
}

#[test]
fn cancellation_signals_before_waiting_for_the_artifact_writer() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "stalled-writer", false);
    let control = store.control(&id).unwrap();
    std::thread::scope(|scope| {
        let (locked, waiting) = std::sync::mpsc::sync_channel(1);
        let (release, held) = std::sync::mpsc::sync_channel(1);
        let writer_store = &store;
        let writer_id = &id;
        let writer = scope.spawn(move || {
            writer_store.with_writer_locked(writer_id, || {
                locked.send(()).unwrap();
                held.recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            })
        });
        waiting
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let canceller = scope.spawn(|| store.cancel(&id));
        let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while control.checkpoint().is_ok() && std::time::Instant::now() < until {
            std::thread::yield_now();
        }
        let cancelled_before_disk = control.checkpoint().is_err();
        assert_eq!(store.status(&id).unwrap().state, JobState::Queued);
        release.send(()).unwrap();
        writer.join().unwrap();
        assert_eq!(
            canceller.join().unwrap().unwrap().state,
            JobState::Cancelled
        );
        assert!(cancelled_before_disk);
    });
    assert_eq!(store.start(&id, 2).unwrap(), Start::Settled);
}

#[test]
fn retained_job_and_aggregate_byte_budgets_refuse_without_eviction() {
    let root = TestRoot::new();
    let store = JobStore::open(
        &root.0,
        Limits {
            max_retained_jobs: 1,
            ..Limits::default()
        },
    )
    .unwrap();
    let id = accepted(&store, "retained", false);
    store.cancel(&id).unwrap();
    assert!(matches!(
        store.accept("next", &request(), false),
        Err(StoreError::Full)
    ));
    assert!(!store.accept("retained", &request(), false).unwrap().created);
    assert_eq!(store.request(&id).unwrap(), request());

    let root = TestRoot::new();
    let store = JobStore::open(
        &root.0,
        Limits {
            max_store_bytes: 2 * 64 * 1024
                + serde_json::to_vec(&request()).unwrap().len() as u64
                + 1,
            ..Limits::default()
        },
    )
    .unwrap();
    let id = accepted(&store, "bytes", false);
    store.start(&id, 2).unwrap();
    assert!(matches!(
        store.append(&id, &[json!({"kind":"prepared_input"})]),
        Err(StoreError::Full)
    ));
    store
        .finish_generation(
            &id,
            StopReason::ExecutionError,
            Counters {
                prompt_tokens: 2,
                ..Counters::default()
            },
            Some(failure()),
        )
        .unwrap();
    assert_eq!(
        store.finalize(&id, Some(failure())).unwrap().state,
        JobState::Failed
    );
}

#[test]
fn acceptance_faults_recover_on_the_correct_side_of_rename() {
    for archive_bytes in [0, 16] {
        for point in [FaultPoint::AcceptanceStaged, FaultPoint::AcceptanceRenamed] {
            let root = TestRoot::new();
            let store = root.open();
            store.fail_once(point);
            assert!(
                store
                    .accept_with_archive("uncertain", &request(), true, archive_bytes)
                    .is_err()
            );
            assert!(
                store
                    .accept_with_archive("uncertain", &request(), true, archive_bytes)
                    .is_err()
            );
            assert!(store.history(None, 10).unwrap().jobs.is_empty());
            drop(store);
            let store = root.open();
            let retry = store
                .accept_with_archive("uncertain", &request(), true, archive_bytes)
                .unwrap();
            if point == FaultPoint::AcceptanceStaged {
                assert!(retry.created);
            } else {
                assert!(!retry.created);
                assert_eq!(retry.status.state, JobState::Interrupted);
            }
            assert_eq!(store.history(None, 10).unwrap().jobs.len(), 1);
            assert!(!fs::read_dir(&root.0).unwrap().any(|child| {
                child
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".pending-")
            }));
        }
    }
}

#[test]
fn renamed_snapshot_failure_keeps_old_visibility_and_recovers_new_watermark() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "rename-fault", true);
    store.start(&id, 2).unwrap();
    let generation = store
        .finish_generation(&id, StopReason::TokenLimit, counters(), None)
        .unwrap()
        .generation;
    store.fail_once(FaultPoint::SnapshotRenamed);
    assert!(store.append(&id, &[readout()]).is_err());
    assert!(store.status(&id).is_err());
    assert!(store.result(&id, None, 10).unwrap().records.is_empty());
    assert!(store.append(&id, &[readout()]).is_err());
    drop(store);
    let store = root.open();
    let status = store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Interrupted);
    assert_eq!(status.generation, generation);
    assert_eq!(store.result(&id, None, 10).unwrap().records.len(), 1);
}

#[test]
fn recovery_syncs_adopted_terminal_snapshot_before_returning_history() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "terminal-rename", false);
    store.start(&id, 2).unwrap();
    store
        .finish_generation(&id, StopReason::TokenLimit, counters(), None)
        .unwrap();
    store.fail_once(FaultPoint::SnapshotRenamed);
    assert!(store.finalize(&id, None).is_err());
    drop(store);
    assert!(JobStore::open_failing_recovery(&root.0, FaultPoint::RecoveryJobSync).is_err());
    let store = root.open();
    let terminal = store.status(&id).unwrap();
    assert_eq!(terminal.state, JobState::Completed);
    drop(store);
    assert_eq!(root.open().status(&id).unwrap(), terminal);
}

#[test]
fn recovery_syncs_root_before_acknowledging_recovered_acceptance() {
    let root = TestRoot::new();
    let store = root.open();
    store.fail_once(FaultPoint::AcceptanceRenamed);
    assert!(store.accept("root-sync", &request(), false).is_err());
    drop(store);
    assert!(JobStore::open_failing_recovery(&root.0, FaultPoint::RecoveryRootSync).is_err());
    let store = root.open();
    let retry = store.accept("root-sync", &request(), false).unwrap();
    assert!(!retry.created);
    assert_eq!(retry.status.state, JobState::Interrupted);
}

#[test]
fn interrupted_recovery_is_idempotent_on_the_next_restart() {
    let root = TestRoot::new();
    let store = root.open();
    let id = accepted(&store, "recovery", false);
    drop(store);
    assert!(JobStore::open_failing_recovery(&root.0, FaultPoint::RecoveryPublished).is_err());
    let store = root.open();
    let first = store.status(&id).unwrap();
    assert_eq!(first.state, JobState::Interrupted);
    drop(store);
    assert_eq!(root.open().status(&id).unwrap(), first);
}
