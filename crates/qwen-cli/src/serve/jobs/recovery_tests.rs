use super::*;
use crate::serve::jobs::tests::TestRoot;
use serde_json::json;

fn saved(store: &JobStore, key: &str) -> String {
    let id = store
        .accept(key, &json!({"key":key}), false)
        .unwrap()
        .status
        .id;
    store.start(&id, 2).unwrap();
    store
        .append(&id, &[json!({"kind":"note","text":"saved result"})])
        .unwrap();
    store
        .finish_generation(
            &id,
            StopReason::TokenLimit,
            Counters {
                prompt_tokens: 2,
                consumed_prompt_tokens: 2,
                sampled_tokens: 1,
                consumed_generated_tokens: 0,
            },
            None,
        )
        .unwrap();
    store.finalize(&id, None).unwrap();
    store.execution_settled(&id);
    id
}

#[test]
fn one_damaged_job_keeps_healthy_history_readable_without_forgetting_accepted_keys() {
    for file in ["status.json", "request.json", "records.jsonl"] {
        let root = TestRoot::new();
        let store = root.open();
        let healthy = saved(&store, "healthy");
        let damaged = saved(&store, "damaged");
        let original = store.status(&healthy).unwrap();
        drop(store);
        let path = root.0.join(&damaged).join(file);
        let backup = fs::read(&path).unwrap();
        fs::write(&path, b"!").unwrap();
        let store = root.open();
        assert_eq!(store.status(&healthy).unwrap(), original);
        assert_eq!(store.request(&healthy).unwrap(), json!({"key":"healthy"}));
        assert_eq!(
            store.result(&healthy, None, 10).unwrap().records[0]["text"],
            "saved result"
        );
        let page = store.history(None, 10).unwrap();
        assert_eq!(page.jobs.len(), 1);
        assert_eq!(page.recovery.unwrap().unavailable_jobs, [damaged.clone()]);
        assert!(matches!(
            store.status(&damaged),
            Err(StoreError::RecoveryRequired)
        ));
        for key in ["fresh", "damaged", "healthy"] {
            assert!(matches!(
                store.lookup(key, &json!({"key":key}), false),
                Err(StoreError::RecoveryRequired)
            ));
            assert!(matches!(
                store.accept(key, &json!({"key":key}), false),
                Err(StoreError::RecoveryRequired)
            ));
        }
        assert!(matches!(
            store.delete(&healthy),
            Err(StoreError::RecoveryRequired)
        ));
        assert_eq!(fs::read(&path).unwrap(), b"!");
        drop(store);
        fs::write(&path, backup).unwrap();
        let recovered = root.open();
        assert!(recovered.recovery_report().is_none());
        assert!(
            !recovered
                .accept("damaged", &json!({"key":"damaged"}), false)
                .unwrap()
                .created
        );
        assert!(
            recovered
                .accept("fresh", &json!({"key":"fresh"}), false)
                .unwrap()
                .created
        );
    }
}

#[test]
fn failed_tombstone_cleanup_is_isolated_without_losing_retry_protection() {
    let root = TestRoot::new();
    let store = root.open();
    let healthy = saved(&store, "healthy");
    let deleted = saved(&store, "deleted");
    store.fail_once(FaultPoint::DeletePayload);
    assert!(store.delete(&deleted).is_err());
    drop(store);
    let store = JobStore::open_failing_recovery(&root.0, FaultPoint::DeletePayload).unwrap();
    assert_eq!(store.history(None, 10).unwrap().jobs[0].id, healthy);
    assert!(matches!(
        store.lookup("deleted", &json!({"key":"deleted"}), false),
        Err(StoreError::RecoveryRequired)
    ));
    drop(store);
    let store = root.open();
    assert!(
        store
            .lookup("deleted", &json!({"key":"deleted"}), false)
            .unwrap()
            .unwrap()
            .deleted
    );
}

#[test]
fn recovery_report_is_bounded_and_symlinked_job_directories_are_not_followed() {
    let root = TestRoot::new();
    drop(root.open());
    for index in 0..65 {
        fs::create_dir(root.0.join(format!("job_{index:x}"))).unwrap();
    }
    let outside = TestRoot::new();
    fs::create_dir(&outside.0).unwrap();
    std::os::unix::fs::symlink(&outside.0, root.0.join("job_ffff")).unwrap();
    let store = root.open();
    let report = store.recovery_report().unwrap();
    assert!(report.read_only);
    assert_eq!(report.unavailable_count, 66);
    assert_eq!(report.unavailable_jobs.len(), 64);
    assert_eq!(fs::read_dir(&outside.0).unwrap().count(), 0);
}

#[test]
fn malformed_non_array_records_and_changed_sequences_are_isolated_even_at_the_same_length() {
    for sequence in [false, true] {
        let root = TestRoot::new();
        let store = root.open();
        let id = saved(&store, "record");
        drop(store);
        let path = root.0.join(&id).join("records.jsonl");
        let mut bytes = fs::read(&path).unwrap();
        let length = bytes.len();
        if sequence {
            let index = bytes
                .windows(7)
                .position(|part| part == b"\"seq\":0")
                .unwrap();
            bytes[index + 6] = b'1';
        } else {
            bytes[0] = b'!';
        }
        fs::write(&path, &bytes).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), length as u64);
        let store = root.open();
        assert_eq!(store.recovery_report().unwrap().unavailable_jobs, [id]);
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}

#[test]
fn non_utf8_job_names_remain_distinct_and_count_toward_inventory_limits() {
    use std::os::unix::ffi::OsStringExt;
    let root = TestRoot::new();
    let mut store = root.open();
    // APFS refuses non-UTF-8 filenames; exercise lossless inventory in memory.
    let mut count = 0;
    for byte in [0x80, 0x81] {
        store.unavailable.insert(std::ffi::OsString::from_vec(
            [b"job_".as_slice(), &[byte]].concat(),
        ));
        assert_eq!(count_job(&mut count, 1).is_ok(), byte == 0x80);
    }
    assert_eq!(count, 2);
    let report = store.recovery_report().unwrap();
    assert_eq!(report.unavailable_count, 2);
    assert_eq!(report.unavailable_jobs.len(), 2);
    assert_ne!(report.unavailable_jobs[0], report.unavailable_jobs[1]);
}
