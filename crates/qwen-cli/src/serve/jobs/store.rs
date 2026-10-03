use super::preview::{RequestPreview, request_preview};
use super::state::*;
use crate::ordinary_executor::ExecutionControl;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_SNAPSHOT_BYTES: usize = 64 * 1024;
const MAX_ERROR_BYTES: usize = 4096;

#[path = "arrays.rs"]
mod arrays;
#[path = "deletion.rs"]
mod deletion;
pub(crate) use arrays::{MAX_ARCHIVE_BYTES, MAX_ARRAY_BYTES};

#[derive(Default)]
struct Faults {
    #[cfg(test)]
    next: Mutex<Option<FaultPoint>>,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaultPoint {
    AcceptanceStaged,
    AcceptanceRenamed,
    RecordPartialWrite,
    RecordSync,
    ArrayPartialWrite,
    ArraySync,
    SnapshotRenamed,
    RecoveryPublished,
    RecoveryJobSync,
    RecoveryRootSync,
    DeletePayload,
}

#[cfg(test)]
impl Faults {
    fn check(&self, point: FaultPoint) -> Result<()> {
        let mut next = self.next.lock().unwrap();
        if *next == Some(point) {
            *next = None;
            return Err(io::Error::other("injected job storage failure").into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) max_active_jobs: usize,
    pub(crate) max_retained_jobs: usize,
    pub(crate) max_retry_identities: usize,
    pub(crate) max_store_bytes: u64,
    pub(crate) max_request_bytes: usize,
    pub(crate) max_record_bytes: usize,
    pub(crate) max_batch_bytes: usize,
    pub(crate) max_job_bytes: u64,
    pub(crate) max_page_bytes: usize,
    pub(crate) max_page_records: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_active_jobs: 32,
            max_retained_jobs: 4096,
            max_retry_identities: 65536,
            max_store_bytes: 64 * 1024 * 1024 * 1024,
            max_request_bytes: 1024 * 1024,
            max_record_bytes: 1024 * 1024,
            max_batch_bytes: 4 * 1024 * 1024,
            max_job_bytes: 4 * 1024 * 1024 * 1024,
            max_page_bytes: 2 * 1024 * 1024,
            max_page_records: 256,
        }
    }
}

#[derive(Debug)]
pub(crate) enum StoreError {
    Invalid(String),
    NotFound,
    Conflict,
    Full,
    Deleted,
    Storage(io::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::NotFound => f.write_str("unknown job"),
            Self::Conflict => f.write_str("idempotency key already names a different request"),
            Self::Full => f.write_str("durable job capacity reached"),
            Self::Deleted => {
                f.write_str("job payloads were deleted; the accepted retry identity is retained")
            }
            Self::Storage(error) => write!(f, "job storage unavailable: {error}"),
        }
    }
}

impl std::error::Error for StoreError {}
impl From<io::Error> for StoreError {
    fn from(error: io::Error) -> Self {
        Self::Storage(error)
    }
}
impl From<serde_json::Error> for StoreError {
    fn from(error: serde_json::Error) -> Self {
        Self::Storage(io::Error::new(io::ErrorKind::InvalidData, error))
    }
}
type Result<T> = std::result::Result<T, StoreError>;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
fn corrupt(message: &str) -> StoreError {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn next_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "job_{time:032x}_{:08x}_{:016x}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    key: String,
    request_hash: String,
    observations_requested: bool,
    status: JobStatus,
    committed_bytes: u64,
    next_seq: u64,
    #[serde(default)]
    archive_reserved_bytes: u64,
    #[serde(default)]
    archive_committed_bytes: u64,
}

struct Entry {
    directory: PathBuf,
    request_preview: RwLock<Option<RequestPreview>>,
    state: RwLock<Snapshot>,
    writer: Mutex<()>,
    fenced: AtomicBool,
    control: ExecutionControl,
    faults: Arc<Faults>,
    runtime: Mutex<super::state::RuntimeStatus>,
    payload: RwLock<()>,
    payload_charge: AtomicU64,
}

impl Entry {
    fn snapshot(&self) -> Snapshot {
        self.state.read().unwrap().clone()
    }

    fn status(&self) -> JobStatus {
        let mut status = self.snapshot().status;
        let runtime = self.runtime.lock().unwrap_or_else(|e| e.into_inner());
        if runtime.publication_error.is_some() {
            status.runtime = Some(runtime.clone());
        }
        status
    }

    fn publication_failed(&self, error: JobError) {
        self.runtime
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .publication_error
            .get_or_insert(error);
        self.control.cancel();
    }

    // Hold writer, not state, across IO: status reads remain CPU-only even if
    // the artifact disk stalls. A failed publication fences further writes.
    fn publish(&self, mut next: Snapshot) -> Result<JobStatus> {
        next.status.revision = next
            .status
            .revision
            .checked_add(1)
            .ok_or_else(|| corrupt("job revision overflow"))?;
        next.status.updated_at_ms = now_ms().max(next.status.updated_at_ms);
        let bytes = snapshot_bytes(&next)?;
        if let Err(error) = replace_snapshot(&self.directory, &bytes, &self.faults) {
            self.fenced.store(true, Ordering::Release);
            self.publication_failed(JobError {
                r#type: "server_error".into(), code: "artifact_write_failed".into(), param: None,
                message: "Publication failed; the displayed durable snapshot may be stale. Restart is required to reconcile disk state.".into(),
            });
            return Err(error);
        }
        let status = next.status.clone();
        *self.state.write().unwrap() = next;
        Ok(status)
    }

    fn writable(&self) -> Result<()> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(io::Error::other(
                "job publication failed; restart required to recover durable state",
            )
            .into());
        }
        Ok(())
    }
}

pub(crate) struct JobStore {
    root: PathBuf,
    limits: Limits,
    entries: RwLock<BTreeMap<String, Arc<Entry>>>,
    acceptance: Mutex<()>,
    fenced: AtomicBool,
    reserved_bytes: AtomicU64,
    faults: Arc<Faults>,
    _lock: File,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Start {
    Running,
    Settled,
}

pub(crate) struct Accepted {
    pub(crate) status: JobStatus,
    pub(crate) created: bool,
}

#[derive(Serialize)]
pub(crate) struct HistoryPage {
    pub(crate) schema_version: u32,
    pub(crate) jobs: Vec<JobStatus>,
    pub(crate) request_previews: BTreeMap<String, Option<RequestPreview>>,
    pub(crate) next_cursor: Option<String>,
    pub(crate) storage: StorageUsage,
}

#[derive(Serialize)]
pub(crate) struct StorageUsage {
    retained_jobs: usize,
    retry_identities: usize,
    reserved_bytes: u64,
    max_retained_jobs: usize,
    max_retry_identities: usize,
    max_store_bytes: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct ResultPage {
    pub(crate) schema_version: u32,
    pub(crate) job_id: String,
    pub(crate) records: Vec<Value>,
    pub(crate) next_cursor: Option<String>,
    pub(crate) complete: bool,
}

impl JobStore {
    pub(crate) fn open(root: &Path, limits: Limits) -> Result<Self> {
        Self::open_inner(root, limits, Arc::new(Faults::default()))
    }

    fn open_inner(root: &Path, limits: Limits, faults: Arc<Faults>) -> Result<Self> {
        if limits.max_active_jobs == 0
            || limits.max_retained_jobs == 0
            || limits.max_retry_identities < limits.max_retained_jobs
            || limits.max_store_bytes < (2 * MAX_SNAPSHOT_BYTES) as u64
            || limits.max_page_records == 0
            || limits.max_record_bytes == 0
            || limits.max_request_bytes == 0
            || limits.max_batch_bytes < limits.max_record_bytes
            || limits.max_page_bytes < limits.max_record_bytes.saturating_add(1024)
            || limits.max_job_bytes < limits.max_batch_bytes as u64
        {
            return Err(invalid("inconsistent job storage limits"));
        }
        match fs::symlink_metadata(root) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => return Err(invalid("job data root must be a directory, not a symlink")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::DirBuilder::new().mode(0o700).create(root)?;
                sync_parent(root)?;
            }
            Err(error) => return Err(error.into()),
        }
        let root = fs::canonicalize(root)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(root.join(".lock"))?;
        if !lock.metadata()?.is_file() {
            return Err(invalid("job lock must be a regular file"));
        }
        // This descriptor remains open for the store lifetime, including all
        // HTTP readers. A second server must never recover live work as dead.
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        let mut entries = BTreeMap::new();
        let mut keys = std::collections::BTreeSet::new();
        let mut reserved_bytes = 0_u64;
        for child in fs::read_dir(&root)? {
            let child = child?;
            let name = child.file_name();
            if name
                .to_str()
                .is_some_and(|name| name.strip_prefix(".pending-").is_some_and(valid_id))
            {
                remove_abandoned_acceptance(&child.path())?;
                continue;
            }
            let Some(id) = name.to_str().filter(|name| name.starts_with("job_")) else {
                continue;
            };
            if !valid_id(id) || !child.file_type()?.is_dir() {
                return Err(corrupt("invalid job directory"));
            }
            if entries.len() >= limits.max_retry_identities {
                return Err(invalid(
                    "retained history exceeds configured job cap; raise the cap to reopen without deleting history",
                ));
            }
            let directory = child.path();
            let mut snapshot: Snapshot =
                read_json(&directory.join("status.json"), MAX_SNAPSHOT_BYTES)?;
            if snapshot.version != 1
                || snapshot.status.schema_version != 1
                || snapshot.status.id != id
                || snapshot.status.runtime.is_some()
                || snapshot.status.result.url != format!("/v1/lens/jobs/{id}/result")
                || snapshot.status.result.complete != snapshot.status.state.terminal()
                || !keys.insert(snapshot.key.clone())
            {
                return Err(corrupt("inconsistent job snapshot"));
            }
            if snapshot.status.deleted {
                deletion::recover(&directory, &snapshot, &faults)?;
                reserved_bytes = reserved_bytes
                    .checked_add((2 * MAX_SNAPSHOT_BYTES) as u64)
                    .ok_or_else(|| corrupt("job storage accounting overflow"))?;
                entries.insert(
                    id.to_owned(),
                    Arc::new(Entry {
                        directory,
                        request_preview: RwLock::new(None),
                        state: RwLock::new(snapshot),
                        writer: Mutex::new(()),
                        fenced: AtomicBool::new(false),
                        control: ExecutionControl::default(),
                        faults: Arc::clone(&faults),
                        runtime: Mutex::new(RuntimeStatus {
                            execution_settled: true,
                            ..Default::default()
                        }),
                        payload: RwLock::new(()),
                        payload_charge: AtomicU64::new(0),
                    }),
                );
                continue;
            }
            // Check the immutable request, but do not retain prompt bodies for
            // the entire history in RAM. Only compact metadata is indexed.
            let request: Value =
                read_json(&directory.join("request.json"), limits.max_request_bytes)?;
            if request_hash(&request)? != snapshot.request_hash {
                return Err(corrupt("job request digest mismatch"));
            }
            arrays::recover(&directory, &snapshot, limits)?;
            let previous_reserved = reserved_bytes;
            reserved_bytes = reserved_bytes
                .checked_add(
                    fs::metadata(directory.join("request.json"))?
                        .len()
                        .checked_add((2 * MAX_SNAPSHOT_BYTES) as u64)
                        .and_then(|bytes| bytes.checked_add(snapshot.committed_bytes))
                        .and_then(|bytes| bytes.checked_add(snapshot.archive_committed_bytes))
                        .ok_or_else(|| corrupt("job storage accounting overflow"))?,
                )
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
                replace_snapshot(&directory, &snapshot_bytes(&snapshot)?, &faults)?;
                #[cfg(test)]
                faults.check(FaultPoint::RecoveryPublished)?;
            }
            // A previous process may have failed after a rename but before its
            // directory sync. Adopt even terminal snapshots durably before
            // exposing recovered history or answering an idempotent retry.
            #[cfg(test)]
            faults.check(FaultPoint::RecoveryJobSync)?;
            File::open(&directory)?.sync_all()?;
            entries.insert(
                id.to_owned(),
                Arc::new(Entry {
                    directory,
                    request_preview: RwLock::new(request_preview(&request)),
                    state: RwLock::new(snapshot),
                    writer: Mutex::new(()),
                    fenced: AtomicBool::new(false),
                    control: ExecutionControl::default(),
                    faults: Arc::clone(&faults),
                    runtime: Mutex::new(RuntimeStatus {
                        execution_settled: true,
                        ..Default::default()
                    }),
                    payload: RwLock::new(()),
                    payload_charge: AtomicU64::new(
                        reserved_bytes - previous_reserved - (2 * MAX_SNAPSHOT_BYTES) as u64,
                    ),
                }),
            );
        }
        if entries
            .values()
            .filter(|entry| !entry.snapshot().status.deleted)
            .count()
            > limits.max_retained_jobs
        {
            return Err(invalid("retained history exceeds configured job cap"));
        }
        #[cfg(test)]
        faults.check(FaultPoint::RecoveryRootSync)?;
        File::open(&root)?.sync_all()?;
        Ok(Self {
            root,
            limits,
            entries: RwLock::new(entries),
            acceptance: Mutex::new(()),
            fenced: AtomicBool::new(false),
            reserved_bytes: AtomicU64::new(reserved_bytes),
            faults,
            _lock: lock,
        })
    }

    fn entry(&self, id: &str) -> Result<Arc<Entry>> {
        self.entries
            .read()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or(StoreError::NotFound)
    }

    pub(crate) fn limits(&self) -> Limits {
        self.limits
    }

    /// Recovery lookup does not depend on the currently loaded model's assets
    /// or capabilities. It never admits new work or consumes queue capacity.
    pub(crate) fn lookup(
        &self,
        key: &str,
        request: &Value,
        observations: bool,
    ) -> Result<Option<JobStatus>> {
        if self.fenced.load(Ordering::Acquire) {
            return Err(io::Error::other("acceptance publication failed; restart required").into());
        }
        let hash = request_hash(request)?;
        for entry in self.entries.read().unwrap().values() {
            let snapshot = entry.snapshot();
            if snapshot.key == key {
                entry.writable()?;
                return if snapshot.request_hash == hash
                    && snapshot.observations_requested == observations
                {
                    Ok(Some(snapshot.status))
                } else {
                    Err(StoreError::Conflict)
                };
            }
        }
        Ok(None)
    }

    /// `request` is the native adapter's validated, normalized specification.
    /// The caller enqueues only after this returns created=true; a lost HTTP
    /// response is recovered with the same key, not by resubmitting inference.
    pub(crate) fn accept(
        &self,
        key: &str,
        request: &Value,
        observations: bool,
    ) -> Result<Accepted> {
        self.accept_with_archive(key, request, observations, 0)
    }

    pub(crate) fn accept_with_archive(
        &self,
        key: &str,
        request: &Value,
        observations: bool,
        archive_bytes: u64,
    ) -> Result<Accepted> {
        if key.is_empty() || key.len() > 256 {
            return Err(invalid("idempotency key must contain 1..256 bytes"));
        }
        let bytes = encode_bounded(&Canonical(request), self.limits.max_request_bytes)?;
        let hash = blake3::hash(&bytes).to_hex().to_string();
        let _acceptance = self.acceptance.lock().unwrap();
        if self.fenced.load(Ordering::Acquire) {
            return Err(io::Error::other("acceptance publication failed; restart required").into());
        }
        let mut active = 0;
        let mut retained = 0;
        for entry in self.entries.read().unwrap().values() {
            let snapshot = entry.snapshot();
            if snapshot.key == key {
                return if snapshot.request_hash == hash
                    && snapshot.observations_requested == observations
                {
                    entry.writable()?;
                    Ok(Accepted {
                        status: snapshot.status,
                        created: false,
                    })
                } else {
                    Err(StoreError::Conflict)
                };
            }
            active += usize::from(!snapshot.status.state.terminal());
            retained += usize::from(!snapshot.status.deleted);
        }
        if active >= self.limits.max_active_jobs
            || retained >= self.limits.max_retained_jobs
            || self.entries.read().unwrap().len() >= self.limits.max_retry_identities
        {
            return Err(StoreError::Full);
        }
        if archive_bytes > MAX_ARCHIVE_BYTES || archive_bytes > self.limits.max_job_bytes {
            return Err(invalid("archive reservation exceeds byte budget"));
        }
        self.reserve_bytes(bytes.len() as u64 + (2 * MAX_SNAPSHOT_BYTES) as u64 + archive_bytes)?;
        let id = next_id();
        let staging = self.root.join(format!(".pending-{id}"));
        fs::DirBuilder::new().mode(0o700).create(&staging)?;
        let directory = self.root.join(&id);
        let snapshot = Snapshot {
            version: 1,
            key: key.into(),
            request_hash: hash,
            observations_requested: observations,
            status: JobStatus::queued(id.clone(), observations, now_ms()),
            committed_bytes: 0,
            next_seq: 0,
            archive_reserved_bytes: archive_bytes,
            archive_committed_bytes: 0,
        };
        let publish = (|| -> Result<()> {
            write_new(&staging.join("request.json"), &bytes)?;
            write_new(&staging.join("records.jsonl"), &[])?;
            if archive_bytes != 0 {
                write_new(&staging.join("arrays.bin"), &[])?;
            }
            write_new(&staging.join("status.json"), &snapshot_bytes(&snapshot)?)?;
            File::open(&staging)?.sync_all()?;
            #[cfg(test)]
            self.faults.check(FaultPoint::AcceptanceStaged)?;
            fs::rename(&staging, &directory)?;
            #[cfg(test)]
            self.faults.check(FaultPoint::AcceptanceRenamed)?;
            File::open(&self.root)?.sync_all()?;
            Ok(())
        })();
        if let Err(error) = publish {
            // Rename may have succeeded before the directory fsync failed.
            // Never create a second execution under an uncertain acceptance.
            self.fenced.store(true, Ordering::Release);
            return Err(error);
        }
        let status = snapshot.status.clone();
        self.entries.write().unwrap().insert(
            id,
            Arc::new(Entry {
                directory,
                request_preview: RwLock::new(request_preview(request)),
                state: RwLock::new(snapshot),
                writer: Mutex::new(()),
                fenced: AtomicBool::new(false),
                control: ExecutionControl::default(),
                faults: Arc::clone(&self.faults),
                runtime: Mutex::default(),
                payload: RwLock::new(()),
                payload_charge: AtomicU64::new(bytes.len() as u64 + archive_bytes),
            }),
        );
        Ok(Accepted {
            status,
            created: true,
        })
    }

    pub(crate) fn status(&self, id: &str) -> Result<JobStatus> {
        let entry = self.entry(id)?;
        Ok(entry.status())
    }

    pub(crate) fn publication_failed(&self, id: &str, error: JobError) {
        if let Ok(entry) = self.entry(id) {
            entry.publication_failed(error);
        }
    }

    pub(crate) fn execution_outcome(&self, id: &str, generation: super::state::RuntimeGeneration) {
        if let Ok(entry) = self.entry(id) {
            entry
                .runtime
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .generation = Some(generation);
        }
    }

    pub(crate) fn execution_settled(&self, id: &str) {
        if let Ok(entry) = self.entry(id) {
            let durable_terminal =
                entry.snapshot().status.state.terminal() && !entry.fenced.load(Ordering::Acquire);
            let mut runtime = entry.runtime.lock().unwrap_or_else(|e| e.into_inner());
            runtime.execution_settled = true;
            if durable_terminal {
                runtime.publication_error = None;
            }
        }
    }

    pub(crate) fn request(&self, id: &str) -> Result<Value> {
        let entry = self.entry(id)?;
        let _payload = entry.payload.read().unwrap();
        if entry.snapshot().status.deleted {
            return Err(StoreError::Deleted);
        }
        read_json(
            &entry.directory.join("request.json"),
            self.limits.max_request_bytes,
        )
    }

    pub(crate) fn control(&self, id: &str) -> Result<ExecutionControl> {
        Ok(self.entry(id)?.control.clone())
    }

    pub(crate) fn history(&self, cursor: Option<&str>, limit: usize) -> Result<HistoryPage> {
        self.check_page_limit(limit)?;
        if cursor.is_some_and(|cursor| !valid_id(cursor)) {
            return Err(invalid("invalid history cursor"));
        }
        let entries = self.entries.read().unwrap();
        let storage = StorageUsage {
            retained_jobs: entries
                .values()
                .filter(|entry| !entry.snapshot().status.deleted)
                .count(),
            retry_identities: entries.len(),
            reserved_bytes: self.reserved_bytes.load(Ordering::Acquire),
            max_retained_jobs: self.limits.max_retained_jobs,
            max_retry_identities: self.limits.max_retry_identities,
            max_store_bytes: self.limits.max_store_bytes,
        };
        let mut selected = entries.iter().rev().filter(|(id, entry)| {
            cursor.is_none_or(|cursor| id.as_str() < cursor)
                && (!entry.snapshot().status.deleted
                    || entry.payload_charge.load(Ordering::Acquire) != 0)
        });
        let mut request_previews = BTreeMap::new();
        let jobs: Vec<_> = selected
            .by_ref()
            .take(limit)
            .map(|(id, entry)| {
                let status = entry.status();
                request_previews.insert(
                    id.clone(),
                    if status.deleted {
                        None
                    } else {
                        entry.request_preview.read().unwrap().clone()
                    },
                );
                status
            })
            .collect();
        let next_cursor = selected
            .next()
            .and_then(|_| jobs.last().map(|job| job.id.clone()));
        Ok(HistoryPage {
            schema_version: 1,
            jobs,
            request_previews,
            next_cursor,
            storage,
        })
    }

    fn update(
        &self,
        id: &str,
        update: impl FnOnce(&mut JobStatus) -> Result<bool>,
    ) -> Result<JobStatus> {
        let entry = self.entry(id)?;
        let _writer = entry.writer.lock().unwrap();
        entry.writable()?;
        let mut next = entry.snapshot();
        if !update(&mut next.status)? {
            return Ok(next.status);
        }
        entry.publish(next)
    }

    pub(crate) fn start(&self, id: &str, prompt_tokens: u64) -> Result<Start> {
        if prompt_tokens == 0 {
            return Err(invalid("prompt must contain tokens"));
        }
        let control = self.control(id)?;
        let status = self.update(id, |status| {
            if status.state.terminal() {
                return Ok(false);
            }
            if status.state != JobState::Queued {
                return Err(invalid("job is not dispatchable"));
            }
            // Cancellation signals before waiting on this transition lock. If
            // start wins the lock, persist that intent rather than a setup error.
            if status.cancel_requested || control.is_cancelled() {
                status.request_cancel();
                return Ok(true);
            }
            status.state = JobState::Running;
            status.generation.state = GenerationState::Running;
            status.generation.phase = Some(Phase::Prefill);
            status.generation.counters.prompt_tokens = prompt_tokens;
            if status.observations.state == ObservationState::Pending {
                status.observations.state = ObservationState::Writing;
            }
            Ok(true)
        })?;
        Ok(if status.state.terminal() {
            Start::Settled
        } else {
            Start::Running
        })
    }

    pub(crate) fn progress(&self, id: &str, phase: Phase, counters: Counters) -> Result<JobStatus> {
        self.update(id, |status| {
            apply_progress(status, phase, counters)?;
            Ok(true)
        })
    }

    pub(crate) fn cancel(&self, id: &str) -> Result<JobStatus> {
        self.cancel_after_signal(id, || {})
    }

    fn cancel_after_signal(&self, id: &str, signalled: impl FnOnce()) -> Result<JobStatus> {
        let entry = self.entry(id)?;
        if !entry.snapshot().status.state.terminal() {
            // Signal immediately, even if the artifact writer is stalled. The
            // returned status acknowledges persistence only after update wins.
            entry.control.cancel();
        }
        signalled();
        let status = self.update(id, |status| {
            if status.state.terminal() || status.cancel_requested {
                return Ok(false);
            }
            status.request_cancel();
            Ok(true)
        })?;
        if status.cancel_requested {
            self.entry(id)?.control.cancel();
        }
        Ok(status)
    }

    /// Delivery or preparation never reached model execution. Cancellation may have already
    /// made this accepted job terminal; preserve that outcome under one lock.
    pub(crate) fn fail_dispatch(&self, id: &str, error: JobError) -> Result<JobStatus> {
        let control = self.control(id)?;
        self.update(id, |status| {
            if status.state.terminal() {
                return Ok(false);
            }
            if status.state != JobState::Queued {
                return Err(invalid("dispatch failure after execution started"));
            }
            if control.is_cancelled() {
                status.request_cancel();
                return Ok(true);
            }
            status.state = JobState::Failed;
            status.generation.state = GenerationState::Failed;
            status.generation.phase = None;
            status.generation.stop_reason = Some(StopReason::ExecutionError);
            status.generation.error = Some(error);
            if status.observations.state == ObservationState::Pending {
                status.observations.state = ObservationState::Partial;
            }
            status.result.complete = true;
            Ok(true)
        })
    }

    pub(crate) fn interrupt(&self, id: &str) -> Result<JobStatus> {
        let control = self.control(id)?;
        control.cancel();
        self.update(id, |status| {
            if status.state.terminal() {
                return Ok(false);
            }
            status.interrupt();
            Ok(true)
        })
    }

    pub(crate) fn finish_generation(
        &self,
        id: &str,
        reason: StopReason,
        counters: Counters,
        error: Option<JobError>,
    ) -> Result<JobStatus> {
        if !matches!(
            (reason, &error),
            (
                StopReason::ExecutionError | StopReason::ServerRestart,
                Some(_)
            ) | (
                StopReason::StopToken | StopReason::TokenLimit | StopReason::Cancelled,
                None
            )
        ) {
            return Err(invalid("inconsistent generation outcome"));
        }
        self.update(id, |status| {
            if status.state.terminal()
                || status.generation.state.terminal()
                || !counters.follows(&status.generation.counters)
                || (status.state == JobState::Queued
                    && !matches!(
                        reason,
                        StopReason::ExecutionError
                            | StopReason::Cancelled
                            | StopReason::ServerRestart
                    ))
            {
                return Err(invalid("generation cannot finish in this state"));
            }
            if matches!(reason, StopReason::StopToken | StopReason::TokenLimit)
                && (counters.consumed_prompt_tokens != counters.prompt_tokens
                    || counters.sampled_tokens == 0
                    || counters.consumed_generated_tokens != counters.sampled_tokens - 1)
            {
                return Err(invalid(
                    "completed generation must retain one unconsumed terminal sample",
                ));
            }
            status.generation.state = match reason {
                StopReason::StopToken | StopReason::TokenLimit => GenerationState::Completed,
                StopReason::Cancelled => GenerationState::Cancelled,
                StopReason::ServerRestart => GenerationState::Interrupted,
                _ => GenerationState::Failed,
            };
            status.generation.phase = None;
            status.generation.counters = counters;
            status.generation.stop_reason = Some(reason);
            status.generation.error = error;
            status.state = JobState::Finalizing;
            Ok(true)
        })
    }

    /// Called after all requested artifact batches have been committed (or a
    /// writer failure has been acknowledged), never merely at generation EOS.
    pub(crate) fn finalize(
        &self,
        id: &str,
        observation_error: Option<JobError>,
    ) -> Result<JobStatus> {
        self.update(id, |status| {
            if status.state != JobState::Finalizing {
                return Err(invalid("job is not finalizing"));
            }
            let failed = observation_error.is_some();
            status.result.error = observation_error.clone();
            if status.observations.state != ObservationState::NotRequested {
                status.observations.state = if failed {
                    if status.observations.committed_records > 0 {
                        ObservationState::Partial
                    } else {
                        ObservationState::Failed
                    }
                } else if status.generation.state == GenerationState::Completed {
                    ObservationState::Complete
                } else {
                    ObservationState::Partial
                };
                status.observations.error = observation_error;
            }
            status.state = if failed {
                JobState::Failed
            } else {
                match status.generation.state {
                    GenerationState::Completed => JobState::Completed,
                    GenerationState::Cancelled => JobState::Cancelled,
                    GenerationState::Interrupted => JobState::Interrupted,
                    _ => JobState::Failed,
                }
            };
            status.result.complete = true;
            Ok(true)
        })
    }

    /// Sequence numbers belong to the store, not the producer. Committed bytes
    /// become visible only after both the log and its new watermark are synced.
    pub(crate) fn append(&self, id: &str, records: &[Value]) -> Result<JobStatus> {
        if records
            .iter()
            .any(|record| record.get("kind").and_then(Value::as_str) == Some("retained_array"))
        {
            return Err(invalid("retained_array descriptors are store-owned"));
        }
        let entry = self.entry(id)?;
        let _writer = entry.writer.lock().unwrap();
        entry.writable()?;
        self.append_locked(&entry, entry.snapshot(), records, None)
    }

    /// Native producer objects are already serialized. Validate without building
    /// their nested Value trees; publish records and progress in one transaction.
    pub(crate) fn append_encoded_progress(
        &self,
        id: &str,
        records: &[&[u8]],
        phase: Phase,
        counters: Counters,
    ) -> Result<JobStatus> {
        let entry = self.entry(id)?;
        let _writer = entry.writer.lock().unwrap();
        entry.writable()?;
        let mut next = entry.snapshot();
        apply_progress(&mut next.status, phase, counters)?;
        let mut batch = Vec::new();
        let mut readouts = 0;
        for &record in records {
            let prefix = format!("{{\"seq\":{},", next.next_seq);
            let size = record
                .len()
                .checked_add(prefix.len())
                .ok_or_else(|| invalid("record size overflow"))?;
            // Replacing '{' subtracts one byte; the JSONL newline adds it back.
            if size > self.limits.max_record_bytes
                || size > self.limits.max_batch_bytes.saturating_sub(batch.len())
            {
                return Err(invalid("record batch exceeds byte budget"));
            }
            readouts += u64::from(encoded_observation(record)?);
            batch.extend_from_slice(prefix.as_bytes());
            batch.extend_from_slice(&record[1..]);
            batch.push(b'\n');
            next.next_seq = next
                .next_seq
                .checked_add(1)
                .ok_or_else(|| invalid("record sequence overflow"))?;
        }
        if batch.is_empty() {
            return entry.publish(next);
        }
        self.publish_batch(&entry, next, batch, readouts, None)
    }

    fn append_locked(
        &self,
        entry: &Entry,
        mut next: Snapshot,
        records: &[Value],
        payload: Option<&[u8]>,
    ) -> Result<JobStatus> {
        if !matches!(next.status.state, JobState::Running | JobState::Finalizing) {
            return Err(invalid("job does not accept records"));
        }
        let mut batch = Vec::new();
        let mut readouts = 0;
        for record in records {
            let record = record
                .as_object()
                .ok_or_else(|| invalid("record must be an object"))?;
            if record.contains_key("seq") || record.get("kind").and_then(Value::as_str).is_none() {
                return Err(invalid("record requires kind and must not supply seq"));
            }
            readouts += u64::from(matches!(
                record.get("kind").and_then(Value::as_str),
                Some("readout" | "residual_pair")
            ));
            #[derive(Serialize)]
            struct SequencedRecord<'a> {
                seq: u64,
                #[serde(flatten)]
                fields: &'a serde_json::Map<String, Value>,
            }
            let bytes = encode_bounded(
                &SequencedRecord {
                    seq: next.next_seq,
                    fields: record,
                },
                self.limits.max_record_bytes - 1,
            )?;
            if bytes.len() + 1 > self.limits.max_record_bytes
                || bytes.len() + 1 > self.limits.max_batch_bytes.saturating_sub(batch.len())
            {
                return Err(invalid("record batch exceeds byte budget"));
            }
            batch.extend_from_slice(&bytes);
            batch.push(b'\n');
            next.next_seq = next
                .next_seq
                .checked_add(1)
                .ok_or_else(|| invalid("record sequence overflow"))?;
        }
        self.publish_batch(entry, next, batch, readouts, payload)
    }

    fn publish_batch(
        &self,
        entry: &Entry,
        mut next: Snapshot,
        batch: Vec<u8>,
        readouts: u64,
        payload: Option<&[u8]>,
    ) -> Result<JobStatus> {
        if readouts > 0 && next.status.observations.state == ObservationState::NotRequested {
            return Err(invalid("readout without requested observations"));
        }
        if batch.is_empty() {
            return Ok(next.status);
        }
        let previous_bytes = next.committed_bytes;
        next.committed_bytes = previous_bytes
            .checked_add(batch.len() as u64)
            .ok_or_else(|| invalid("result byte count overflow"))?;
        if next
            .committed_bytes
            .checked_add(next.archive_reserved_bytes)
            .is_none_or(|bytes| bytes > self.limits.max_job_bytes)
        {
            return Err(StoreError::Full);
        }
        self.reserve_bytes(batch.len() as u64)?;
        entry
            .payload_charge
            .fetch_add(batch.len() as u64, Ordering::AcqRel);
        let write = (|| -> Result<()> {
            if let Some(bytes) = payload {
                let mut file = open_regular(&entry.directory.join("arrays.bin"), true)?;
                file.seek(SeekFrom::Start(
                    next.archive_committed_bytes - bytes.len() as u64,
                ))?;
                #[cfg(test)]
                if let Err(error) = self.faults.check(FaultPoint::ArrayPartialWrite) {
                    file.write_all(&bytes[..bytes.len() / 2])?;
                    return Err(error);
                }
                file.write_all(bytes)?;
                #[cfg(test)]
                self.faults.check(FaultPoint::ArraySync)?;
                file.sync_all()?;
            }
            let mut file = open_regular(&entry.directory.join("records.jsonl"), true)?;
            file.seek(SeekFrom::Start(previous_bytes))?;
            #[cfg(test)]
            if let Err(error) = self.faults.check(FaultPoint::RecordPartialWrite) {
                file.write_all(&batch[..batch.len() / 2])?;
                return Err(error);
            }
            file.write_all(&batch)?;
            #[cfg(test)]
            self.faults.check(FaultPoint::RecordSync)?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(error) = write {
            // No watermark has moved. Preserve the generation outcome while
            // allowing a CPU finalizer to publish an observation failure.
            return Err(error);
        }
        next.status.result.available = true;
        next.status.observations.committed_records += readouts;
        entry.publish(next)
    }

    fn check_page_limit(&self, limit: usize) -> Result<()> {
        if limit == 0 || limit > self.limits.max_page_records {
            return Err(invalid("invalid page record limit"));
        }
        Ok(())
    }

    fn reserve_bytes(&self, bytes: u64) -> Result<()> {
        self.reserved_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= self.limits.max_store_bytes)
            })
            .map_err(|_| StoreError::Full)?;
        // Failed writes retain their reservation until recovery truncates
        // uncommitted tails and recomputes usage. No automatic eviction.
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fail_once(&self, point: FaultPoint) {
        *self.faults.next.lock().unwrap() = Some(point);
    }

    #[cfg(test)]
    pub(super) fn open_failing_recovery(root: &Path, point: FaultPoint) -> Result<Self> {
        let faults = Arc::new(Faults::default());
        *faults.next.lock().unwrap() = Some(point);
        Self::open_inner(root, Limits::default(), faults)
    }

    #[cfg(test)]
    pub(crate) fn with_writer_locked<T>(&self, id: &str, action: impl FnOnce() -> T) -> T {
        let entry = self.entry(id).unwrap();
        let _writer = entry.writer.lock().unwrap();
        action()
    }

    #[cfg(test)]
    pub(crate) fn cancel_paused(&self, id: &str, signalled: impl FnOnce()) -> Result<JobStatus> {
        self.cancel_after_signal(id, signalled)
    }

    #[cfg(test)]
    pub(crate) fn with_history_locked(&self, action: impl FnOnce()) {
        let _entries = self.entries.write().unwrap();
        action();
    }

    pub(crate) fn result(
        &self,
        id: &str,
        cursor: Option<&str>,
        limit: usize,
    ) -> Result<ResultPage> {
        self.check_page_limit(limit)?;
        let entry = self.entry(id)?;
        let _payload = entry.payload.read().unwrap();
        let snapshot = entry.snapshot();
        if snapshot.status.deleted {
            return Err(StoreError::Deleted);
        }
        let (mut offset, mut seq) = parse_cursor(id, cursor)?;
        if offset > snapshot.committed_bytes
            || seq > snapshot.next_seq
            || (offset == snapshot.committed_bytes && seq != snapshot.next_seq)
            || ((offset == 0) != (seq == 0))
        {
            return Err(invalid("result cursor outside committed records"));
        }
        let mut file = open_regular(&entry.directory.join("records.jsonl"), false)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset - 1))?;
            let mut byte = [0];
            file.read_exact(&mut byte)?;
            if byte[0] != b'\n' {
                return Err(invalid("result cursor is not a record boundary"));
            }
        } else {
            file.seek(SeekFrom::Start(0))?;
        }
        let mut reader = BufReader::new(file.take(snapshot.committed_bytes - offset));
        let mut records = Vec::new();
        let mut bytes = 1024; // Envelope and cursor reserve, included in page budget.
        while offset < snapshot.committed_bytes && records.len() < limit {
            let mut line = Vec::new();
            (&mut reader)
                .take(self.limits.max_record_bytes as u64 + 1)
                .read_until(b'\n', &mut line)?;
            if line.len() > self.limits.max_record_bytes || line.last() != Some(&b'\n') {
                return Err(corrupt("invalid committed result record"));
            }
            if bytes + line.len() > self.limits.max_page_bytes {
                break;
            }
            let record: Value = serde_json::from_slice(&line)?;
            if record.get("seq").and_then(Value::as_u64) != Some(seq) {
                return Err(invalid("result cursor sequence mismatch"));
            }
            offset += line.len() as u64;
            seq += 1;
            bytes += line.len();
            records.push(record);
        }
        let complete = offset == snapshot.committed_bytes && snapshot.status.result.complete;
        let next_cursor = (!complete).then(|| make_cursor(id, offset, seq));
        Ok(ResultPage {
            schema_version: 1,
            job_id: id.into(),
            records,
            next_cursor,
            complete,
        })
    }
}

fn valid_id(id: &str) -> bool {
    id.starts_with("job_")
        && (5..=80).contains(&id.len())
        && id[4..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() || byte == b'_')
}
fn request_hash(request: &Value) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(&Canonical(request))?)
        .to_hex()
        .to_string())
}
struct Canonical<'a>(&'a Value);
impl Serialize for Canonical<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self.0 {
            Value::Object(object) => {
                let sorted: BTreeMap<_, _> = object.iter().collect();
                let mut map = serializer.serialize_map(Some(sorted.len()))?;
                for (key, value) in sorted {
                    map.serialize_entry(key, &Canonical(value))?;
                }
                map.end()
            }
            Value::Array(array) => {
                let mut seq = serializer.serialize_seq(Some(array.len()))?;
                for value in array {
                    seq.serialize_element(&Canonical(value))?;
                }
                seq.end()
            }
            value => value.serialize(serializer),
        }
    }
}

fn encode_bounded(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    struct Buffer {
        bytes: Vec<u8>,
        limit: usize,
        exceeded: bool,
    }
    impl Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                self.exceeded = true;
                return Err(io::Error::other("serialized value exceeds byte budget"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut buffer = Buffer {
        bytes: Vec::new(),
        limit,
        exceeded: false,
    };
    let encoded = serde_json::to_writer(&mut buffer, value);
    if buffer.exceeded {
        return Err(invalid("serialized value exceeds byte budget"));
    }
    encoded?;
    Ok(buffer.bytes)
}

fn make_cursor(id: &str, offset: u64, seq: u64) -> String {
    format!(
        "v1-{}-{offset:x}-{seq:x}",
        &blake3::hash(id.as_bytes()).to_hex()[..16]
    )
}
fn parse_cursor(id: &str, cursor: Option<&str>) -> Result<(u64, u64)> {
    let Some(cursor) = cursor else {
        return Ok((0, 0));
    };
    if cursor.len() > 64 {
        return Err(invalid("invalid result cursor"));
    }
    let parts: Vec<_> = cursor.split('-').collect();
    if parts.len() != 4
        || parts[0] != "v1"
        || parts[1] != &blake3::hash(id.as_bytes()).to_hex()[..16]
    {
        return Err(invalid("invalid result cursor"));
    }
    let offset = u64::from_str_radix(parts[2], 16).map_err(|_| invalid("invalid result cursor"))?;
    let seq = u64::from_str_radix(parts[3], 16).map_err(|_| invalid("invalid result cursor"))?;
    Ok((offset, seq))
}

fn open_regular(path: &Path, write: bool) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(write)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(corrupt("expected a regular job file"));
    }
    Ok(file)
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path, limit: usize) -> Result<T> {
    let file = open_regular(path, false)?;
    if file.metadata()?.len() > limit as u64 {
        return Err(corrupt("job metadata exceeds byte limit"));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(corrupt("job metadata exceeds byte limit"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}
fn snapshot_bytes(snapshot: &Snapshot) -> Result<Vec<u8>> {
    for error in [
        &snapshot.status.generation.error,
        &snapshot.status.observations.error,
        &snapshot.status.result.error,
    ]
    .into_iter()
    .flatten()
    {
        encode_bounded(error, MAX_ERROR_BYTES)?;
    }
    encode_bounded(snapshot, MAX_SNAPSHOT_BYTES / 2)
}

fn remove_abandoned_acceptance(directory: &Path) -> Result<()> {
    if !fs::symlink_metadata(directory)?.is_dir() {
        return Err(corrupt("invalid abandoned acceptance directory"));
    }
    let children = fs::read_dir(directory)?.collect::<std::result::Result<Vec<_>, _>>()?;
    for child in &children {
        if !matches!(
            child.file_name().to_str(),
            Some("request.json" | "status.json" | "records.jsonl" | "arrays.bin")
        ) || !child.file_type()?.is_file()
        {
            return Err(corrupt(
                "unexpected file in abandoned acceptance; retained for inspection",
            ));
        }
    }
    for child in children {
        fs::remove_file(child.path())?;
    }
    fs::remove_dir(directory)?;
    sync_parent(directory)
}

pub(crate) fn check_progress(
    previous_phase: Option<Phase>,
    previous: &Counters,
    phase: Phase,
    counters: &Counters,
) -> Result<()> {
    if !counters.follows(previous)
        || (previous_phase == Some(Phase::Decode) && phase == Phase::Prefill)
        || (phase == Phase::Decode && counters.consumed_prompt_tokens != counters.prompt_tokens)
        || (phase == Phase::Prefill && counters.sampled_tokens != 0)
    {
        return Err(invalid("invalid generation progress"));
    }
    Ok(())
}

fn apply_progress(status: &mut JobStatus, phase: Phase, counters: Counters) -> Result<()> {
    if status.generation.state != GenerationState::Running {
        return Err(invalid("invalid generation progress"));
    }
    check_progress(
        status.generation.phase,
        &status.generation.counters,
        phase,
        &counters,
    )?;
    status.generation.phase = Some(phase);
    status.generation.counters = counters;
    Ok(())
}

fn encoded_observation(bytes: &[u8]) -> Result<bool> {
    fn present<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<bool, D::Error> {
        serde::de::IgnoredAny::deserialize(d).map(|_| true)
    }
    #[derive(Deserialize)]
    struct Envelope {
        kind: String,
        #[serde(default, deserialize_with = "present")]
        seq: bool,
    }
    if bytes.first() != Some(&b'{')
        || bytes.last() != Some(&b'}')
        || bytes.iter().any(|b| matches!(b, b'\n' | b'\r'))
    {
        return Err(invalid("serialized record must be one JSONL object"));
    }
    let envelope: Envelope = serde_json::from_slice(bytes)?;
    if envelope.seq || envelope.kind == "retained_array" {
        return Err(invalid(
            "record sequence and array descriptors are store-owned",
        ));
    }
    Ok(matches!(
        envelope.kind.as_str(),
        "readout" | "residual_pair"
    ))
}

#[cfg(test)]
#[test]
fn encoded_sequence_overflow_never_publishes_partial_batch() {
    let root = super::tests::TestRoot::new();
    let store = root.open();
    let id = store
        .accept("overflow", &serde_json::json!({}), false)
        .unwrap()
        .status
        .id;
    store.start(&id, 1).unwrap();
    store.entry(&id).unwrap().state.write().unwrap().next_seq = u64::MAX;
    assert!(
        store
            .append_encoded_progress(
                &id,
                &[br#"{"kind":"test"}"#],
                Phase::Prefill,
                Counters {
                    prompt_tokens: 1,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(
        fs::metadata(root.0.join(&id).join("records.jsonl"))
            .unwrap()
            .len(),
        0
    );
    assert_eq!(store.entry(&id).unwrap().snapshot().next_seq, u64::MAX);
}

fn replace_snapshot(directory: &Path, bytes: &[u8], _faults: &Faults) -> Result<()> {
    let temporary = directory.join(".status.next");
    // Only one store process and one per-job writer can reach this path. A
    // regular orphan from a crash is safe to overwrite, never follow a symlink.
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&temporary)?;
    if !file.metadata()?.is_file() {
        return Err(corrupt("invalid job snapshot staging file"));
    }
    file.set_len(0)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, directory.join("status.json"))?;
    #[cfg(test)]
    _faults.check(FaultPoint::SnapshotRenamed)?;
    File::open(directory)?.sync_all()?;
    Ok(())
}
