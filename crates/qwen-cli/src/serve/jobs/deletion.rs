use super::*;

const PAYLOADS: [&str; 3] = ["request.json", "records.jsonl", "arrays.bin"];

fn cleanup(directory: &Path, faults: &Faults) -> Result<()> {
    for name in PAYLOADS {
        let path = directory.join(name);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() => {
                fs::remove_file(path)?;
                #[cfg(test)]
                faults.check(FaultPoint::DeletePayload)?;
            }
            Ok(_) => return Err(corrupt("deleted job payload is not a regular file")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    let _ = faults;
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub(super) fn recover(directory: &Path, snapshot: &Snapshot, faults: &Faults) -> Result<()> {
    if !snapshot.status.state.terminal()
        || snapshot.status.result.available
        || snapshot.committed_bytes != 0
        || snapshot.next_seq != 0
        || snapshot.archive_reserved_bytes != 0
        || snapshot.archive_committed_bytes != 0
    {
        return Err(corrupt("invalid deleted job snapshot"));
    }
    File::open(directory)?.sync_all()?;
    cleanup(directory, faults)
}

impl JobStore {
    pub(crate) fn delete(&self, id: &str) -> Result<JobStatus> {
        self.delete_inner(id, || {})
    }

    fn delete_inner(&self, id: &str, waiting: impl FnOnce()) -> Result<JobStatus> {
        let _acceptance = self.acceptance.lock().unwrap();
        let entry = self.entry(id)?;
        let _writer = entry.writer.lock().unwrap();
        let _payload = match entry.payload.try_write() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                waiting();
                entry.payload.write().unwrap()
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(corrupt("job payload lock poisoned"));
            }
        };
        entry.writable()?;
        let mut next = entry.snapshot();
        if !next.status.deleted {
            if !next.status.state.terminal()
                || !entry
                    .runtime
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .execution_settled
            {
                return Err(invalid(
                    "only durable terminal jobs with settled execution can be deleted",
                ));
            }
            next.status.deleted = true;
            next.status.result.available = false;
            next.status.observations.committed_records = 0;
            next.committed_bytes = 0;
            next.next_seq = 0;
            next.archive_reserved_bytes = 0;
            next.archive_committed_bytes = 0;
            entry.publish(next)?;
            *entry.request_preview.write().unwrap() = None;
        }
        cleanup(&entry.directory, &entry.faults)?;
        // Failed or partial cleanup stays conservatively charged until completion.
        self.reserved_bytes.fetch_sub(
            entry.payload_charge.swap(0, Ordering::AcqRel),
            Ordering::AcqRel,
        );
        Ok(entry.status())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::jobs::tests::TestRoot;
    use serde_json::json;

    fn terminal(store: &JobStore, key: &str) -> String {
        let id = store
            .accept_with_archive(key, &json!({"key":key}), true, 8)
            .unwrap()
            .status
            .id;
        store.start(&id, 2).unwrap();
        store
            .append_array(
                &id,
                json!({"key":"source","quantity":"source_residual"}),
                &[0; 8],
            )
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
        id
    }

    #[test]
    fn deletion_requires_settlement_reclaims_payloads_and_preserves_retry_identity() {
        let root = TestRoot::new();
        let limits = Limits {
            max_retained_jobs: 1,
            max_retry_identities: 2,
            ..Limits::default()
        };
        let store = JobStore::open(&root.0, limits).unwrap();
        let id = terminal(&store, "first");
        assert!(store.delete(&id).is_err());
        store.execution_settled(&id);
        let deleted = store.delete(&id).unwrap();
        assert!(deleted.deleted);
        assert!(!deleted.result.available);
        assert_eq!(store.delete(&id).unwrap(), deleted);
        for name in PAYLOADS {
            assert!(!root.0.join(&id).join(name).exists());
        }
        assert!(matches!(store.request(&id), Err(StoreError::Deleted)));
        assert!(matches!(
            store.result(&id, None, 10),
            Err(StoreError::Deleted)
        ));
        assert!(matches!(store.array(&id, 0), Err(StoreError::Deleted)));
        assert!(store.history(None, 10).unwrap().jobs.is_empty());
        assert_eq!(
            store.reserved_bytes.load(Ordering::Acquire),
            (2 * MAX_SNAPSHOT_BYTES) as u64
        );
        assert!(
            store
                .lookup("first", &json!({"key":"first"}), true)
                .unwrap()
                .unwrap()
                .deleted
        );
        assert!(matches!(
            store.lookup("first", &json!({"key":"changed"}), true),
            Err(StoreError::Conflict)
        ));
        assert!(matches!(
            store.lookup("first", &json!({"key":"first"}), false),
            Err(StoreError::Conflict)
        ));
        let second = terminal(&store, "second");
        store.execution_settled(&second);
        store.delete(&second).unwrap();
        assert!(matches!(
            store.accept("third", &json!({}), false),
            Err(StoreError::Full)
        ));
        drop(store);
        let recovered = JobStore::open(&root.0, limits).unwrap();
        assert_eq!(
            recovered
                .lookup("first", &json!({"key":"first"}), true)
                .unwrap()
                .unwrap(),
            deleted
        );
    }

    #[test]
    fn deletion_faults_recover_without_forgetting_keys_or_undercharging_payloads() {
        for fault in [FaultPoint::SnapshotRenamed, FaultPoint::DeletePayload] {
            let root = TestRoot::new();
            let store = root.open();
            let id = terminal(&store, "fault");
            store.execution_settled(&id);
            let charged = store.reserved_bytes.load(Ordering::Acquire);
            store.fail_once(fault);
            assert!(store.delete(&id).is_err());
            assert_eq!(store.reserved_bytes.load(Ordering::Acquire), charged);
            assert!(root.0.join(&id).join("arrays.bin").exists());
            if fault == FaultPoint::SnapshotRenamed {
                assert!(root.0.join(&id).join("request.json").exists());
                assert!(
                    store
                        .lookup("fault", &json!({"key":"fault"}), true)
                        .is_err()
                );
            } else {
                assert!(store.history(None, 10).unwrap().jobs[0].deleted);
                assert!(
                    store
                        .lookup("fault", &json!({"key":"fault"}), true)
                        .unwrap()
                        .unwrap()
                        .deleted
                );
            }
            drop(store);
            let recovered = root.open();
            assert!(recovered.status(&id).unwrap().deleted);
            assert!(recovered.history(None, 10).unwrap().jobs.is_empty());
            assert_eq!(
                recovered.reserved_bytes.load(Ordering::Acquire),
                (2 * MAX_SNAPSHOT_BYTES) as u64
            );
            for name in PAYLOADS {
                assert!(!root.0.join(&id).join(name).exists());
            }
        }
    }

    #[test]
    fn deletion_waits_for_an_existing_reader_and_never_follows_payload_symlinks() {
        let root = TestRoot::new();
        let store = Arc::new(root.open());
        let id = terminal(&store, "reader");
        store.execution_settled(&id);
        let entry = store.entry(&id).unwrap();
        let reader = entry.payload.read().unwrap();
        let other = store.clone();
        let other_id = id.clone();
        let (sent, received) = std::sync::mpsc::sync_channel(1);
        let deletion = std::thread::spawn(move || {
            other.delete_inner(&other_id, || {
                sent.send(()).unwrap();
            })
        });
        received
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(!entry.snapshot().status.deleted);
        drop(reader);
        assert!(deletion.join().unwrap().unwrap().deleted);

        let id = terminal(&store, "symlink");
        store.execution_settled(&id);
        let victim = root.0.join("unrelated");
        fs::write(&victim, b"keep").unwrap();
        let payload = root.0.join(&id).join("request.json");
        fs::remove_file(&payload).unwrap();
        std::os::unix::fs::symlink(&victim, &payload).unwrap();
        assert!(store.delete(&id).is_err());
        assert_eq!(fs::read(victim).unwrap(), b"keep");
    }
}
