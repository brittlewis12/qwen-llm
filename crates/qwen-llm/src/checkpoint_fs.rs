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

use std::fs::{File, FileTimes, OpenOptions};
use std::io;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::SystemTime;

pub(crate) const TEMP_PREFIX: &str = ".tmp-";
const LOCK_FILE: &str = "store.lock";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
}

impl StoreNamespace {
    pub(crate) fn new(root: impl Into<PathBuf>, version: &'static str) -> Self {
        Self {
            root: root.into(),
            version,
            ready: Arc::new(AtomicBool::new(false)),
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

    fn lock(&self, operation: libc::c_int) -> Result<StoreLock, CheckpointFsError> {
        let initialize = !self.ready.load(Ordering::Acquire);
        if initialize {
            self.ensure_namespace()?;
        }
        let namespace = self.namespace_root();
        ensure_real_directory(&namespace)?;
        let path = namespace.join(LOCK_FILE);
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
                self.namespace_root().join(LOCK_FILE),
            ));
        }
        if initialize {
            sync_directory(&namespace)?;
            self.ready.store(true, Ordering::Release);
        }
        flock_retry(&file, operation)?;
        Ok(StoreLock { file })
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

    /// Create (and durably record) the blobs root and one compat directory.
    pub(crate) fn ensure_blob_dir(&self, blob_dir: &Path) -> Result<(), CheckpointFsError> {
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
        Ok(())
    }

    /// Open a managed blob for reading under a shared lock.
    pub(crate) fn open_candidate(
        &self,
        path: &Path,
    ) -> Result<Option<BlobLease>, CheckpointFsError> {
        let _lock = self.lock_shared()?;
        open_blob_nofollow(path)
    }

    /// Unlink a managed blob only while it is still the leased inode.
    pub(crate) fn remove_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointFsError> {
        let _lock = self.lock_exclusive()?;
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
        let _lock = self.lock_exclusive()?;
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
        "{TEMP_PREFIX}{}-{nonce}-{sequence}-{}",
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
