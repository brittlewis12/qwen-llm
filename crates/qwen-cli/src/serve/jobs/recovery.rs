use super::*;

pub(super) fn count_job(count: &mut usize, limit: usize) -> Result<()> {
    *count = count
        .checked_add(1)
        .ok_or_else(|| corrupt("job inventory count overflow"))?;
    if *count > limit {
        return Err(invalid("history exceeds configured retry identity cap"));
    }
    Ok(())
}

pub(super) fn load(
    child: &fs::DirEntry,
    id: &str,
    limits: Limits,
    faults: &Arc<Faults>,
    keys: &mut std::collections::BTreeSet<String>,
) -> Result<Entry> {
    if !valid_id(id) || !child.file_type()?.is_dir() {
        return Err(corrupt("invalid job directory"));
    }
    let directory = child.path();
    let mut snapshot: Snapshot = read_json(&directory.join("status.json"), MAX_SNAPSHOT_BYTES)?;
    if snapshot.version != 1
        || snapshot.status.schema_version != 1
        || snapshot.status.id != id
        || snapshot.status.runtime.is_some()
        || snapshot.status.result.url != format!("/v1/lens/jobs/{id}/result")
        || snapshot.status.result.complete != snapshot.status.state.terminal()
        || snapshot.key.is_empty()
        || snapshot.key.len() > 256
        || snapshot.request_hash.len() != 64
        || !snapshot
            .request_hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || !keys.insert(snapshot.key.clone())
    {
        return Err(corrupt("inconsistent job snapshot"));
    }

    let (preview, charge) = if snapshot.status.deleted {
        deletion::recover(&directory, &snapshot, faults)?;
        (None, 0)
    } else {
        let request: Value = read_json(&directory.join("request.json"), limits.max_request_bytes)?;
        if request_hash(&request)? != snapshot.request_hash {
            return Err(corrupt("job request digest mismatch"));
        }
        arrays::recover(&directory, &snapshot, limits)?;
        let charge = fs::metadata(directory.join("request.json"))?
            .len()
            .checked_add(snapshot.committed_bytes)
            .and_then(|bytes| bytes.checked_add(snapshot.archive_committed_bytes))
            .ok_or_else(|| corrupt("job storage accounting overflow"))?;
        let records = open_regular(&directory.join("records.jsonl"), true)?;
        if records.metadata()?.len() < snapshot.committed_bytes {
            return Err(corrupt("committed result bytes are missing"));
        }
        records.set_len(snapshot.committed_bytes)?;
        records.sync_all()?;
        if !snapshot.status.state.terminal() {
            snapshot.status.interrupt();
            snapshot.status.revision = snapshot
                .status
                .revision
                .checked_add(1)
                .ok_or_else(|| corrupt("job revision overflow"))?;
            snapshot.status.updated_at_ms = now_ms().max(snapshot.status.updated_at_ms);
            replace_snapshot(&directory, &snapshot_bytes(&snapshot)?, faults)?;
            #[cfg(test)]
            faults.check(FaultPoint::RecoveryPublished)?;
        }
        // Adopt even terminal snapshots before exposing history or retry results.
        #[cfg(test)]
        faults.check(FaultPoint::RecoveryJobSync)?;
        File::open(&directory)?.sync_all()?;
        (request_preview(&request), charge)
    };
    Ok(Entry {
        directory,
        request_preview: RwLock::new(preview),
        state: RwLock::new(snapshot),
        writer: Mutex::new(()),
        fenced: AtomicBool::new(false),
        control: ExecutionControl::default(),
        faults: Arc::clone(faults),
        runtime: Mutex::new(RuntimeStatus {
            execution_settled: true,
            ..Default::default()
        }),
        payload: RwLock::new(()),
        payload_charge: AtomicU64::new(charge),
    })
}

#[derive(Clone, Serialize)]
pub(crate) struct RecoveryReport {
    pub(crate) read_only: bool,
    pub(crate) unavailable_count: usize,
    pub(crate) unavailable_jobs: Vec<String>,
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod tests;
