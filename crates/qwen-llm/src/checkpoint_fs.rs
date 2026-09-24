//! Family-agnostic filesystem mechanics for durable checkpoint blob stores.
//!
//! `checkpoint_store` (Qwen runtime snapshots) and
//! `deepseek_v4_checkpoint_store` (DeepSeek V4 causal snapshots) share one
//! namespace discipline: a versioned private root holding a `store.lock`
//! advisory-flock file, an `identity` cache, and `blobs/<compat hex64>/`
//! directories of immutable hard-linked records staged as private temp files.
//! Everything here is family-blind — locking, staged-file stamp validation,
//! LRU byte-budget eviction, foreign-entry rejection, and hex/name plumbing.
//! Family policy (codec, prefix keys, candidate ordering, dedupe equality,
//! telemetry) stays in the two store modules.
//!
//! The shared error carries only the variants both stores expose verbatim;
//! each store converts with `From` so its public error surface is unchanged.

use std::ffi::OsStr;
use std::fs::{File, FileTimes, OpenOptions};
use std::io;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::SystemTime;

pub(crate) const TEMP_PREFIX: &str = ".tmp-";
const LOCK_FILE: &str = "store.lock";
/// Serializes publishers (never readers) so a free-space check and the
/// staging write it admits are atomic with respect to other publishers.
const WRITER_LOCK_FILE: &str = "writer.lock";
const LOCKED_TEMP_VERSION: &str = "l1";
const MAX_STAGING_EXAMINED_PER_PUBLISH: usize = 256;
pub(crate) const MAX_STAGING_SCAVENGE_PER_PUBLISH: usize = 64;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Longest a lookup waits for the namespace lock.
pub(crate) const LOOKUP_LOCK_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

/// Free space a publish leaves on the volume: the cache yields to the rest
/// of the disk rather than write it full.
pub(crate) const VOLUME_FREE_RESERVE_BYTES: u64 = 2 << 30;

/// Available bytes on the volume holding `path` (or its nearest existing
/// ancestor); `None` when unreadable.
pub fn volume_free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let existing = path.ancestors().find(|candidate| candidate.exists())?;
    let path = std::ffi::CString::new(existing.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `path` is NUL-terminated and `stat` is a valid out-pointer.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: statvfs succeeded and initialized the struct.
    let stat = unsafe { stat.assume_init() };
    // Field widths differ across platforms (u32 vs u64 block counts).
    #[allow(clippy::unnecessary_cast)]
    (stat.f_bavail as u64).checked_mul(stat.f_frsize as u64)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointFsError {
    #[error("checkpoint store I/O: {0}")]
    Io(#[from] io::Error),
    #[error("foreign non-regular entry at managed checkpoint key: {0}")]
    ForeignEntryAtKey(PathBuf),
    #[error("checkpoint managed-byte accounting overflow")]
    ManagedBytesOverflow,
    #[error("checkpoint blob size {blob_bytes} exceeds managed budget {max_managed_blob_bytes}")]
    OversizedBlob {
        blob_bytes: u64,
        max_managed_blob_bytes: u64,
    },
    #[error("checkpoint store mutated namespace before {operation} failed: {source}")]
    PostMutationIo {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("checkpoint staged metadata failed: {0}")]
    StagedMetadata(&'static str),
    #[error("checkpoint touch lost an eviction or replacement race")]
    TouchLostRace,
}

/// Versioned private store namespace: `<root>/<version>/{store.lock, identity,
/// blobs/}`. Owns the one-time namespace bootstrap and the whole-store
/// advisory flock.
#[derive(Clone, Debug)]
pub(crate) struct StoreNamespace {
    root: PathBuf,
    version: &'static str,
    ready: Arc<AtomicBool>,
    /// Which sibling compatibility directory the next publish sweeps.
    sibling_sweep_turn: Arc<std::sync::atomic::AtomicUsize>,
    /// Test seam: volume free bytes to report instead of `statvfs`.
    #[cfg(test)]
    pub(crate) test_free_bytes: Arc<std::sync::Mutex<Option<u64>>>,
}

impl StoreNamespace {
    pub(crate) fn new(root: impl Into<PathBuf>, version: &'static str) -> Self {
        Self {
            root: root.into(),
            version,
            ready: Arc::new(AtomicBool::new(false)),
            sibling_sweep_turn: Arc::default(),
            #[cfg(test)]
            test_free_bytes: Arc::default(),
        }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn namespace_root(&self) -> PathBuf {
        self.root.join(self.version)
    }

    pub(crate) fn identity_root(&self) -> PathBuf {
        self.namespace_root().join("identity")
    }

    pub(crate) fn blobs_root(&self) -> PathBuf {
        self.namespace_root().join("blobs")
    }

    pub(crate) fn blob_dir(&self, compatibility_id: &[u8; 32]) -> PathBuf {
        self.blobs_root().join(hex(compatibility_id))
    }

    pub(crate) fn lock_shared(&self) -> Result<StoreLock, CheckpointFsError> {
        self.lock(libc::LOCK_SH)
    }

    pub(crate) fn lock_exclusive(&self) -> Result<StoreLock, CheckpointFsError> {
        self.lock(libc::LOCK_EX)
    }

    /// Shared lock for lookups, which run on request paths: give up after
    /// [`LOOKUP_LOCK_WAIT`] rather than stall behind a publisher's scan, evict
    /// and fsync in this or another process. Busy is an `Io(WouldBlock)`
    /// error, which callers log and treat as a miss.
    pub(crate) fn lock_shared_for_lookup(&self) -> Result<StoreLock, CheckpointFsError> {
        let file = self.open_lock_file(LOCK_FILE)?;
        let started = std::time::Instant::now();
        loop {
            match flock_retry(&file, libc::LOCK_SH | libc::LOCK_NB) {
                Ok(()) => return Ok(StoreLock { file }),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    if started.elapsed() >= LOOKUP_LOCK_WAIT {
                        return Err(CheckpointFsError::Io(io::Error::new(
                            io::ErrorKind::WouldBlock,
                            format!(
                                "checkpoint namespace busy for {LOOKUP_LOCK_WAIT:?} (a writer holds it)"
                            ),
                        )));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Make room to stage a `record_bytes` record while leaving
    /// [`VOLUME_FREE_RESERVE_BYTES`] free. The budget is sized from free space
    /// at startup; when other activity has since filled the volume, evict this
    /// namespace's oldest blobs first, and fail with `StorageFull` if that is
    /// not enough, before writing anything. Returns the bytes evicted. Call
    /// with [`Self::lock_writer`] held so no other publisher stages between
    /// the check and the write.
    pub(crate) fn ensure_volume_space(
        &self,
        record_bytes: u64,
        is_managed_name: impl Fn(&std::ffi::OsStr) -> bool,
    ) -> Result<u64, CheckpointFsError> {
        #[cfg(test)]
        if let Some(free) = *self.test_free_bytes.lock().unwrap() {
            return self.ensure_volume_space_with(record_bytes, is_managed_name, || Some(free));
        }
        self.ensure_volume_space_with(record_bytes, is_managed_name, || {
            volume_free_bytes(&self.root)
        })
    }

    pub(crate) fn ensure_volume_space_with(
        &self,
        record_bytes: u64,
        is_managed_name: impl Fn(&std::ffi::OsStr) -> bool,
        mut free_bytes: impl FnMut() -> Option<u64>,
    ) -> Result<u64, CheckpointFsError> {
        let needed = record_bytes.saturating_add(VOLUME_FREE_RESERVE_BYTES);
        let unreadable = || {
            CheckpointFsError::Io(io::Error::other(
                "volume free space is unreadable; not staging a record that could fill it",
            ))
        };
        let free = free_bytes().ok_or_else(unreadable)?;
        if free >= needed {
            return Ok(0);
        }
        let (free, evicted_bytes) = {
            let _lock = self.lock_exclusive()?;
            // Re-read under the lock: a lookup's lease may have released space.
            let free = free_bytes().ok_or_else(unreadable)?;
            let scan = scan_managed_blobs(&self.blobs_root(), is_managed_name)?;
            let total = scan.total_bytes;
            let shortfall = needed.saturating_sub(free).min(total);
            (free, evict_to_fit(scan, shortfall, total, None)?.1)
        };
        let free_after = free_bytes().unwrap_or(0);
        if evicted_bytes > 0 {
            tracing::warn!(
                "checkpoint store: volume low on space; evicted {evicted_bytes} bytes of oldest blobs \
                 (free {free} -> {free_after}) to stage a {record_bytes}-byte record"
            );
        }
        if free_after < needed {
            return Err(CheckpointFsError::Io(io::Error::new(
                io::ErrorKind::StorageFull,
                format!(
                    "volume has {free_after} bytes free; a {record_bytes}-byte record needs \
                     {needed} with the {VOLUME_FREE_RESERVE_BYTES}-byte reserve"
                ),
            )));
        }
        Ok(evicted_bytes)
    }

    fn lock(&self, operation: libc::c_int) -> Result<StoreLock, CheckpointFsError> {
        let file = self.open_lock_file(LOCK_FILE)?;
        flock_retry(&file, operation)?;
        Ok(StoreLock { file })
    }

    /// Exclusive namespace lock if free right now; `None` under contention.
    /// Lookup maintenance (LRU touch, corrupt removal) uses this so a request
    /// path never waits on a publisher.
    fn try_lock_exclusive(&self) -> Result<Option<StoreLock>, CheckpointFsError> {
        let file = self.open_lock_file(LOCK_FILE)?;
        match flock_retry(&file, libc::LOCK_EX | libc::LOCK_NB) {
            Ok(()) => Ok(Some(StoreLock { file })),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Held by a publisher from its free-space check through publication.
    pub(crate) fn lock_writer(&self) -> Result<StoreLock, CheckpointFsError> {
        let file = self.open_lock_file(WRITER_LOCK_FILE)?;
        flock_retry(&file, libc::LOCK_EX)?;
        Ok(StoreLock { file })
    }

    fn open_lock_file(&self, name: &str) -> Result<File, CheckpointFsError> {
        let initialize = !self.ready.load(Ordering::Acquire);
        if initialize {
            self.ensure_namespace()?;
        }
        let namespace = self.namespace_root();
        ensure_real_directory(&namespace)?;
        let path = namespace.join(name);
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = options.open(path)?;
        if !file.metadata()?.file_type().is_file() {
            return Err(CheckpointFsError::ForeignEntryAtKey(
                self.namespace_root().join(name),
            ));
        }
        if initialize {
            sync_directory(&namespace)?;
            self.ready.store(true, Ordering::Release);
        }
        Ok(file)
    }

    fn ensure_namespace(&self) -> Result<(), CheckpointFsError> {
        create_directory_tree_synced(&self.root)?;
        let namespace = self.namespace_root();
        let namespace_created = create_directory(&namespace)?;
        ensure_real_directory(&namespace)?;
        if namespace_created {
            sync_directory(&self.root)?;
        }
        Ok(())
    }

    /// Create (and durably record) the blobs root and one compat directory,
    /// then reclaim any abandoned staging inodes in this family namespace.
    pub(crate) fn ensure_blob_dir(
        &self,
        blob_dir: &Path,
    ) -> Result<StagingCleanupReport, CheckpointFsError> {
        let _lock = self.lock_exclusive()?;
        let blobs_root = self.blobs_root();
        let blobs_created = create_directory(&blobs_root)?;
        ensure_real_directory(&blobs_root)?;
        if blobs_created {
            sync_directory(&self.namespace_root())?;
        }
        let blob_dir_created = create_directory(blob_dir)?;
        ensure_real_directory(blob_dir)?;
        if blob_dir_created {
            sync_directory(&blobs_root)?;
        }
        let mut report = scavenge_staging_files_locked(blob_dir)?;
        // A crash strands staging files in whichever model's directory was
        // publishing; sweep one sibling per publish, rotating, so switching
        // models does not leave them outside every budget while this lock's
        // hold stays bounded. Best effort: a sibling's error never fails this
        // publish.
        let mut siblings: Vec<PathBuf> = std::fs::read_dir(&blobs_root)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                    .map(|entry| entry.path())
                    .filter(|path| path != blob_dir)
                    .collect()
            })
            .unwrap_or_default();
        if !siblings.is_empty() {
            siblings.sort();
            let turn = self.sibling_sweep_turn.fetch_add(1, Ordering::Relaxed);
            if let Ok(swept) = scavenge_staging_files_locked(&siblings[turn % siblings.len()]) {
                report.absorb(swept);
            }
        }
        Ok(report)
    }

    /// Create and lock one staging file without exposing an unlocked managed
    /// temp name to a concurrent scavenger.
    pub(crate) fn create_staging_file(
        &self,
        path: &Path,
        readable: bool,
    ) -> Result<File, CheckpointFsError> {
        let _lock = self.lock_shared()?;
        ensure_real_directory(path.parent().expect("staging file parent"))?;
        let mut options = OpenOptions::new();
        options
            .read(readable)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = options.open(path)?;
        if let Err(error) = flock_retry(&file, libc::LOCK_EX | libc::LOCK_NB) {
            let descriptor =
                file.metadata()
                    .map_err(|source| CheckpointFsError::PostMutationIo {
                        operation: "inspect staging file after lock failure",
                        source,
                    })?;
            if let Some(current) =
                metadata_nofollow(path).map_err(|source| CheckpointFsError::PostMutationIo {
                    operation: "inspect staging path after lock failure",
                    source,
                })?
                && descriptor.dev() == current.dev()
                && descriptor.ino() == current.ino()
            {
                std::fs::remove_file(path).map_err(|source| CheckpointFsError::PostMutationIo {
                    operation: "remove staging file after lock failure",
                    source,
                })?;
                sync_directory(path.parent().expect("staging file parent")).map_err(|source| {
                    CheckpointFsError::PostMutationIo {
                        operation: "sync staging directory after lock failure",
                        source,
                    }
                })?;
            }
            return Err(error.into());
        }
        Ok(file)
    }

    /// Open a managed blob for reading under a shared lock.
    pub(crate) fn open_candidate(
        &self,
        path: &Path,
    ) -> Result<Option<BlobLease>, CheckpointFsError> {
        let _lock = self.lock_shared()?;
        open_blob_nofollow(path)
    }

    /// [`Self::open_candidate`] under the bounded lookup lock.
    pub(crate) fn open_candidate_for_lookup(
        &self,
        path: &Path,
    ) -> Result<Option<BlobLease>, CheckpointFsError> {
        let _lock = self.lock_shared_for_lookup()?;
        open_blob_nofollow(path)
    }

    /// Unlink a managed blob only while it is still the leased inode.
    pub(crate) fn remove_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointFsError> {
        let _lock = self.lock_exclusive()?;
        Self::remove_if_same_inode_locked(path, lease)
    }

    /// Lookup-path removal of a proven-corrupt blob: skipped (left for a
    /// later lookup or publish) when a publisher holds the namespace.
    pub(crate) fn remove_if_same_inode_for_lookup(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointFsError> {
        let Some(_lock) = self.try_lock_exclusive()? else {
            return Ok(false);
        };
        Self::remove_if_same_inode_locked(path, lease)
    }

    fn remove_if_same_inode_locked(
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointFsError> {
        let Some(metadata) = metadata_nofollow(path)? else {
            return Ok(false);
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Ok(false);
        }
        if metadata.dev() != lease.dev || metadata.ino() != lease.ino {
            return Ok(false);
        }
        std::fs::remove_file(path)?;
        sync_directory(path.parent().expect("blob parent")).map_err(|source| {
            CheckpointFsError::PostMutationIo {
                operation: "sync repaired blob directory",
                source,
            }
        })?;
        Ok(true)
    }

    /// LRU-touch a managed blob only while it is still the leased inode.
    pub(crate) fn touch_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<(), CheckpointFsError> {
        // An LRU touch is optional: skip it rather than wait on a publisher.
        let Some(_lock) = self.try_lock_exclusive()? else {
            return Err(CheckpointFsError::TouchLostRace);
        };
        let metadata = metadata_nofollow(path)?.ok_or(CheckpointFsError::TouchLostRace)?;
        if metadata.dev() != lease.dev || metadata.ino() != lease.ino {
            return Err(CheckpointFsError::TouchLostRace);
        }
        lease
            .file
            .set_times(FileTimes::new().set_modified(SystemTime::now()))?;
        Ok(())
    }
}

/// Whole-store advisory lock; released on drop.
pub(crate) struct StoreLock {
    file: File,
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = flock_retry(&self.file, libc::LOCK_UN);
    }
}

/// Read lease on one blob inode: decode from the open descriptor stays valid
/// even if the name is concurrently evicted or replaced.
pub(crate) struct BlobLease {
    pub(crate) file: File,
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) size: u64,
}

/// Anti-substitution stamp of one file description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FileStamp {
    pub(crate) regular: bool,
    pub(crate) len: u64,
    pub(crate) mode: u32,
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) nlink: u64,
}

impl FileStamp {
    pub(crate) fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            regular: metadata.file_type().is_file(),
            len: metadata.len(),
            mode: metadata.mode() & 0o7777,
            dev: metadata.dev(),
            ino: metadata.ino(),
            nlink: metadata.nlink(),
        }
    }
}

pub(crate) struct ManagedBlob {
    pub(crate) path: PathBuf,
    pub(crate) size: u64,
    pub(crate) modified: SystemTime,
}

pub(crate) struct ManagedScan {
    pub(crate) blobs: Vec<ManagedBlob>,
    pub(crate) total_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct StagingCleanupReport {
    pub(crate) examined_entries: usize,
    pub(crate) removed_entries: usize,
    pub(crate) reclaimed_bytes: u64,
    pub(crate) live_entries: usize,
    pub(crate) legacy_entries: usize,
    pub(crate) foreign_entries: usize,
    pub(crate) truncated: bool,
}

impl StagingCleanupReport {
    fn absorb(&mut self, other: Self) {
        self.examined_entries += other.examined_entries;
        self.removed_entries += other.removed_entries;
        self.reclaimed_bytes = self.reclaimed_bytes.saturating_add(other.reclaimed_bytes);
        self.live_entries += other.live_entries;
        self.legacy_entries += other.legacy_entries;
        self.foreign_entries += other.foreign_entries;
        self.truncated |= other.truncated;
    }
}

pub(crate) fn flock_retry(file: &File, operation: libc::c_int) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    loop {
        let result = unsafe { libc::flock(file.as_raw_fd(), operation) };
        if result == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

pub(crate) fn metadata_nofollow(path: &Path) -> io::Result<Option<std::fs::Metadata>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

pub(crate) fn path_exists_nofollow(path: &Path) -> io::Result<bool> {
    Ok(metadata_nofollow(path)?.is_some())
}

pub(crate) fn ensure_real_directory(path: &Path) -> Result<(), CheckpointFsError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CheckpointFsError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(())
}

pub(crate) fn require_real_directory_if_exists(path: &Path) -> Result<bool, CheckpointFsError> {
    let Some(metadata) = metadata_nofollow(path)? else {
        return Ok(false);
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CheckpointFsError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(true)
}

pub(crate) fn create_directory(path: &Path) -> io::Result<bool> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

pub(crate) fn create_directory_tree_synced(path: &Path) -> Result<(), CheckpointFsError> {
    if require_real_directory_if_exists(path)? {
        return Ok(());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent != path {
        create_directory_tree_synced(parent)?;
    }
    let created = create_directory(path)?;
    ensure_real_directory(path)?;
    if created {
        sync_directory(parent)?;
    }
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

pub(crate) fn open_blob_nofollow(path: &Path) -> Result<Option<BlobLease>, CheckpointFsError> {
    if let Some(metadata) = metadata_nofollow(path)?
        && (metadata.file_type().is_symlink() || !metadata.file_type().is_file())
    {
        return Err(CheckpointFsError::ForeignEntryAtKey(path.to_path_buf()));
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
            return Err(CheckpointFsError::ForeignEntryAtKey(path.to_path_buf()));
        }
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(CheckpointFsError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(Some(BlobLease {
        file,
        dev: metadata.dev(),
        ino: metadata.ino(),
        size: metadata.len(),
    }))
}

pub(crate) fn unique_temp_path(blob_dir: &Path, digest: &[u8; 32]) -> PathBuf {
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    blob_dir.join(format!(
        "{TEMP_PREFIX}{LOCKED_TEMP_VERSION}-{}-{nonce}-{sequence}-{}",
        std::process::id(),
        &hex(digest)[..16]
    ))
}

pub(crate) fn validate_staged_stamp(
    actual: &FileStamp,
    expected_len: u64,
    expected_nlink: u64,
    expected_identity: Option<&FileStamp>,
) -> Result<(), CheckpointFsError> {
    if !actual.regular {
        return Err(CheckpointFsError::StagedMetadata("not a regular file"));
    }
    if actual.len != expected_len {
        return Err(CheckpointFsError::StagedMetadata("length mismatch"));
    }
    if actual.mode != 0o600 {
        return Err(CheckpointFsError::StagedMetadata("mode mismatch"));
    }
    if actual.nlink != expected_nlink {
        return Err(CheckpointFsError::StagedMetadata("link-count mismatch"));
    }
    if expected_identity
        .is_some_and(|expected| actual.dev != expected.dev || actual.ino != expected.ino)
    {
        return Err(CheckpointFsError::StagedMetadata("inode mismatch"));
    }
    Ok(())
}

pub(crate) fn validate_post_link_stamp(
    actual: &FileStamp,
    expected_len: u64,
    expected_identity: &FileStamp,
) -> Result<(), CheckpointFsError> {
    validate_staged_stamp(actual, expected_len, 2, Some(expected_identity))
}

/// Inventory every recognized blob under `root`, where recognition of final
/// entry names is the caller's (family-specific) predicate.
pub(crate) fn scan_managed_blobs(
    root: &Path,
    is_managed_name: impl Fn(&std::ffi::OsStr) -> bool,
) -> Result<ManagedScan, CheckpointFsError> {
    let mut blobs = Vec::new();
    let mut total_bytes = 0u64;
    if !require_real_directory_if_exists(root)? {
        return Ok(ManagedScan { blobs, total_bytes });
    }
    let compat_dirs = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(ManagedScan { blobs, total_bytes });
        }
        Err(error) => return Err(error.into()),
    };
    for compat_dir in compat_dirs {
        let compat_dir = compat_dir?;
        let name = compat_dir.file_name();
        if !is_lower_hex_64(&name) {
            continue;
        }
        let metadata = std::fs::symlink_metadata(compat_dir.path())?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(CheckpointFsError::ForeignEntryAtKey(compat_dir.path()));
        }
        for entry in std::fs::read_dir(compat_dir.path())? {
            let entry = entry?;
            if !is_managed_name(&entry.file_name()) {
                continue;
            }
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                continue;
            }
            total_bytes = total_bytes
                .checked_add(metadata.len())
                .ok_or(CheckpointFsError::ManagedBytesOverflow)?;
            blobs.push(ManagedBlob {
                path: entry.path(),
                size: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }
    Ok(ManagedScan { blobs, total_bytes })
}

/// Return after finding the first recognized regular blob. This is an
/// existence hint only; publication and eviction still require a full scan.
/// After a hit, remaining compatibility directories are still validated so a
/// substituted managed directory cannot be hidden by traversal order.
pub(crate) fn has_managed_blob(
    root: &Path,
    is_managed_name: impl Fn(&OsStr) -> bool,
) -> Result<bool, CheckpointFsError> {
    if !require_real_directory_if_exists(root)? {
        return Ok(false);
    }
    let compat_dirs = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let mut found = false;
    for compat_dir in compat_dirs {
        let compat_dir = compat_dir?;
        if !is_lower_hex_64(&compat_dir.file_name()) {
            continue;
        }
        let metadata = std::fs::symlink_metadata(compat_dir.path())?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(CheckpointFsError::ForeignEntryAtKey(compat_dir.path()));
        }
        if found {
            continue;
        }
        for entry in std::fs::read_dir(compat_dir.path())? {
            let entry = entry?;
            if !is_managed_name(&entry.file_name()) {
                continue;
            }
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if !metadata.file_type().is_symlink() && metadata.file_type().is_file() {
                found = true;
                break;
            }
        }
    }
    Ok(found)
}

fn scavenge_staging_files_locked(
    blob_dir: &Path,
) -> Result<StagingCleanupReport, CheckpointFsError> {
    let mut report = StagingCleanupReport::default();
    if !require_real_directory_if_exists(blob_dir)? {
        return Ok(report);
    }
    let effective_uid = unsafe { libc::geteuid() };
    for entry in std::fs::read_dir(blob_dir)? {
        if report.removed_entries == MAX_STAGING_SCAVENGE_PER_PUBLISH {
            report.truncated = true;
            break;
        }
        let entry = entry.map_err(|source| staging_cleanup_io_error(blob_dir, &report, source))?;
        if !entry.file_name().to_string_lossy().starts_with(TEMP_PREFIX) {
            continue;
        }
        if report.examined_entries == MAX_STAGING_EXAMINED_PER_PUBLISH {
            report.truncated = true;
            break;
        }
        report.examined_entries += 1;
        let Some(staging_kind) = parse_staging_name(&entry.file_name()) else {
            report.foreign_entries += 1;
            continue;
        };
        if staging_kind == StagingKind::LegacyUnlocked {
            // An unlocked legacy file can belong to an older live binary; age
            // and PID cannot prove abandonment, so automatic cleanup leaves it.
            report.legacy_entries += 1;
            continue;
        }
        let path = entry.path();
        let path_metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(staging_cleanup_io_error(blob_dir, &report, error)),
        };
        if path_metadata.file_type().is_symlink()
            || !path_metadata.file_type().is_file()
            || path_metadata.uid() != effective_uid
            || path_metadata.mode() & 0o7777 != 0o600
            || !(1..=2).contains(&path_metadata.nlink())
        {
            report.foreign_entries += 1;
            continue;
        }
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = match options.open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
                report.foreign_entries += 1;
                continue;
            }
            Err(error) => return Err(staging_cleanup_io_error(blob_dir, &report, error)),
        };
        if !try_flock_exclusive(&file)
            .map_err(|source| staging_cleanup_io_error(blob_dir, &report, source))?
        {
            report.live_entries += 1;
            continue;
        }
        let descriptor_metadata = file
            .metadata()
            .map_err(|source| staging_cleanup_io_error(blob_dir, &report, source))?;
        let Some(current_metadata) = metadata_nofollow(&path)
            .map_err(|source| staging_cleanup_io_error(blob_dir, &report, source))?
        else {
            continue;
        };
        if !descriptor_metadata.file_type().is_file()
            || descriptor_metadata.uid() != effective_uid
            || descriptor_metadata.mode() & 0o7777 != 0o600
            || !(1..=2).contains(&descriptor_metadata.nlink())
            || current_metadata.file_type().is_symlink()
            || !current_metadata.file_type().is_file()
            || descriptor_metadata.dev() != current_metadata.dev()
            || descriptor_metadata.ino() != current_metadata.ino()
        {
            report.foreign_entries += 1;
            continue;
        }
        let reclaimed_bytes = if descriptor_metadata.nlink() == 1 {
            let allocated_bytes = descriptor_metadata
                .blocks()
                .checked_mul(512)
                .ok_or(CheckpointFsError::ManagedBytesOverflow)?;
            match report.reclaimed_bytes.checked_add(allocated_bytes) {
                Some(bytes) => bytes,
                None => {
                    sync_staging_cleanup_mutation(blob_dir, &report)?;
                    return Err(CheckpointFsError::ManagedBytesOverflow);
                }
            }
        } else {
            report.reclaimed_bytes
        };
        if let Err(source) = std::fs::remove_file(&path) {
            return Err(staging_cleanup_io_error(blob_dir, &report, source));
        }
        report.removed_entries += 1;
        report.reclaimed_bytes = reclaimed_bytes;
    }
    sync_staging_cleanup_mutation(blob_dir, &report)?;
    Ok(report)
}

fn sync_staging_cleanup_mutation(
    blob_dir: &Path,
    report: &StagingCleanupReport,
) -> Result<(), CheckpointFsError> {
    if report.removed_entries == 0 {
        return Ok(());
    }
    sync_directory(blob_dir).map_err(|source| CheckpointFsError::PostMutationIo {
        operation: "sync scavenged staging directory",
        source,
    })
}

fn staging_cleanup_io_error(
    blob_dir: &Path,
    report: &StagingCleanupReport,
    source: io::Error,
) -> CheckpointFsError {
    if report.removed_entries > 0 {
        if let Err(sync_error) = sync_directory(blob_dir) {
            return CheckpointFsError::PostMutationIo {
                operation: "sync staging directory after cleanup failure",
                source: sync_error,
            };
        }
        CheckpointFsError::PostMutationIo {
            operation: "continue staging cleanup after prior unlink",
            source,
        }
    } else {
        source.into()
    }
}

fn try_flock_exclusive(file: &File) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    loop {
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        if error.kind() == io::ErrorKind::WouldBlock || error.raw_os_error() == Some(libc::EAGAIN) {
            return Ok(false);
        }
        return Err(error);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StagingKind {
    LegacyUnlocked,
    LockedV1,
}

fn parse_staging_name(name: &OsStr) -> Option<StagingKind> {
    let name = name.to_str()?;
    let rest = name.strip_prefix(TEMP_PREFIX)?;
    let (kind, rest) = match rest.strip_prefix(&format!("{LOCKED_TEMP_VERSION}-")) {
        Some(rest) => (StagingKind::LockedV1, rest),
        None => (StagingKind::LegacyUnlocked, rest),
    };
    let mut parts = rest.split('-');
    let (Some(pid), Some(nonce), Some(sequence), Some(digest)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return None;
    };
    (parts.next().is_none()
        && canonical_decimal(pid, |value| {
            value.parse::<u32>().ok().filter(|&value| value > 0)
        })
        && canonical_decimal(nonce, |value| value.parse::<u128>().ok())
        && canonical_decimal(sequence, |value| value.parse::<u64>().ok())
        && digest.len() == 16
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then_some(kind)
}

fn canonical_decimal<T>(value: &str, parse: impl FnOnce(&str) -> Option<T>) -> bool {
    !value.is_empty()
        && (value.len() == 1 || !value.starts_with('0'))
        && value.bytes().all(|byte| byte.is_ascii_digit())
        && parse(value).is_some()
}

/// Evict least-recently-touched blobs until `incoming_bytes` fits the budget.
pub(crate) fn evict_to_fit(
    mut scan: ManagedScan,
    incoming_bytes: u64,
    max_bytes: u64,
    protected: Option<&Path>,
) -> Result<(usize, u64, u64), CheckpointFsError> {
    scan.blobs.sort_by(|a, b| {
        a.modified
            .cmp(&b.modified)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut evicted_entries = 0usize;
    let mut evicted_bytes = 0u64;
    let mut affected_directories = std::collections::BTreeSet::new();
    for blob in scan.blobs {
        if scan
            .total_bytes
            .checked_add(incoming_bytes)
            .ok_or(CheckpointFsError::ManagedBytesOverflow)?
            <= max_bytes
        {
            break;
        }
        if protected.is_some_and(|path| path == blob.path) {
            continue;
        }
        if let Err(source) = std::fs::remove_file(&blob.path) {
            if evicted_entries > 0 {
                return Err(CheckpointFsError::PostMutationIo {
                    operation: "continue eviction after prior unlink",
                    source,
                });
            }
            return Err(source.into());
        }
        affected_directories.insert(
            blob.path
                .parent()
                .expect("managed blob has parent")
                .to_path_buf(),
        );
        scan.total_bytes -= blob.size;
        evicted_entries += 1;
        evicted_bytes = evicted_bytes
            .checked_add(blob.size)
            .ok_or(CheckpointFsError::ManagedBytesOverflow)?;
    }
    if scan
        .total_bytes
        .checked_add(incoming_bytes)
        .ok_or(CheckpointFsError::ManagedBytesOverflow)?
        > max_bytes
    {
        return Err(CheckpointFsError::OversizedBlob {
            blob_bytes: incoming_bytes,
            max_managed_blob_bytes: max_bytes,
        });
    }
    for directory in affected_directories {
        sync_directory(&directory).map_err(|source| CheckpointFsError::PostMutationIo {
            operation: "sync evicted blob directory",
            source,
        })?;
    }
    Ok((evicted_entries, evicted_bytes, scan.total_bytes))
}

pub(crate) fn is_lower_hex_64(value: &std::ffi::OsStr) -> bool {
    value.to_str().is_some_and(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

pub(crate) fn parse_hex_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        out[index] = high << 4 | low;
    }
    if hex(&out) != value {
        return None;
    }
    Some(out)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
