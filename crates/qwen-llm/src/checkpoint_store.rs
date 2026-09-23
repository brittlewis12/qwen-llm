//! Bounded, catalog-free durable checkpoint blob store.
//!
//! The byte budget covers recognized, linked blob files only. Open-but-unlinked
//! restore leases, staging files, identity entries, and foreign files are not a
//! physical free-space guarantee.
//!
//! The configured root is a private store namespace. The implementation rejects
//! substituted managed directories and final entries, but does not defend every
//! parent-component lookup against a malicious process mutating the namespace
//! concurrently.
//!
//! Checkpoints are disposable cache state, not an authoritative transaction log.
//! Corruption repair or budget eviction may complete before a later operation
//! returns an error; callers must remain correct after any cache entry disappears.

use crate::checkpoint_codec::{
    EncodedSnapshot, SnapshotCodecConstraints, SnapshotCodecError, decode_snapshot, encode_snapshot,
};
use crate::checkpoint_fs::{
    BlobLease, CheckpointFsError, FileStamp, StagingCleanupReport, StoreNamespace, evict_to_fit,
    has_managed_blob, hex, metadata_nofollow, parse_hex_32, path_exists_nofollow,
    require_real_directory_if_exists, scan_managed_blobs, sync_directory, unique_temp_path,
    validate_post_link_stamp, validate_staged_stamp,
};
use crate::checkpoint_identity::CheckpointIdentityCache;
use crate::metal_forward::{SessionSnapshot, SnapshotIdentity};
use std::collections::{BTreeSet, HashMap};
use std::fs::FileTimes;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

const NAMESPACE_VERSION: &str = "v1";
const PREFIX_KEY_DOMAIN: &[u8] = b"qwen-checkpoint-prefix-key-v1\0";
const BLOB_EXTENSION: &str = "qcp";
const MAX_PUBLICATION_ATTEMPTS: usize = 8;

#[derive(Clone, Debug)]
pub struct DurableCheckpointStore {
    namespace: StoreNamespace,
    max_managed_blob_bytes: u64,
    staged_integrity: StagedIntegrityMode,
    staged_integrity_explicit: bool,
}

impl DurableCheckpointStore {
    pub fn new(root: impl Into<PathBuf>, max_managed_blob_bytes: u64) -> Self {
        Self {
            namespace: StoreNamespace::new(root, NAMESPACE_VERSION),
            max_managed_blob_bytes,
            staged_integrity: StagedIntegrityMode::Decode,
            staged_integrity_explicit: false,
        }
    }

    pub fn with_staged_integrity(
        root: impl Into<PathBuf>,
        max_managed_blob_bytes: u64,
        staged_integrity: StagedIntegrityMode,
    ) -> Self {
        Self {
            namespace: StoreNamespace::new(root, NAMESPACE_VERSION),
            max_managed_blob_bytes,
            staged_integrity,
            staged_integrity_explicit: true,
        }
    }

    pub fn root(&self) -> &Path {
        self.namespace.root()
    }

    pub fn max_managed_blob_bytes(&self) -> u64 {
        self.max_managed_blob_bytes
    }

    pub fn staged_integrity_mode(&self) -> StagedIntegrityMode {
        self.staged_integrity
    }

    pub fn staged_integrity_is_explicit(&self) -> bool {
        self.staged_integrity_explicit
    }

    pub fn identity_cache(&self) -> CheckpointIdentityCache {
        CheckpointIdentityCache::new(self.namespace.identity_root())
    }

    /// Cheap global emptiness probe for callers that can skip strong identity
    /// resolution when no checkpoint blob could possibly match.
    pub fn has_managed_blobs(&self) -> Result<bool, CheckpointStoreError> {
        let _lock = self.namespace.lock_shared()?;
        Ok(has_managed_blob(&self.blobs_root(), is_managed_blob_name)?)
    }

    pub fn publish(
        &self,
        context: StoreContext<'_>,
        snapshot: &SessionSnapshot,
    ) -> Result<PublishReport, CheckpointStoreError> {
        let mode = SnapshotMode::from_snapshot(snapshot);
        let matched_len = snapshot.matched_prefix_len();
        let digest = snapshot_prefix_key(context.compatibility_id, snapshot);
        let blob_dir = self.blob_dir(context.compatibility_id);
        let staging_cleanup = self.ensure_blob_dir(&blob_dir)?;
        let final_path = blob_dir.join(blob_name(matched_len, mode, &digest));
        let temp_path = unique_temp_path(&blob_dir, &digest);
        let staged = match self.encode_staged(&temp_path, context, snapshot) {
            Ok(staged) => staged,
            Err(error) => {
                let _ = std::fs::remove_file(&temp_path);
                return Err(error);
            }
        };
        if staged.encoded.record_bytes > self.max_managed_blob_bytes {
            let _ = std::fs::remove_file(&temp_path);
            return Err(CheckpointStoreError::OversizedBlob {
                blob_bytes: staged.encoded.record_bytes,
                max_managed_blob_bytes: self.max_managed_blob_bytes,
            });
        }

        let result = self.publish_staged(
            context,
            snapshot,
            mode,
            digest,
            &final_path,
            &staged,
            staging_cleanup,
        );
        let _ = std::fs::remove_file(&staged.path);
        result
    }

    pub fn lookup(
        &self,
        context: StoreContext<'_>,
        request_tokens: &[i32],
    ) -> Result<LookupReport, CheckpointStoreError> {
        self.lookup_filtered(context, request_tokens, |_, _| true)
    }

    /// [`Self::lookup`], consulting `admit(matched_len, blob_bytes)` before
    /// decoding each candidate (longest first). A rejected candidate is
    /// skipped without reading it, so callers can require a longer match
    /// than they already hold or refuse records they cannot afford to load.
    pub fn lookup_filtered(
        &self,
        context: StoreContext<'_>,
        request_tokens: &[i32],
        mut admit: impl FnMut(usize, u64) -> bool,
    ) -> Result<LookupReport, CheckpointStoreError> {
        if request_tokens.is_empty() {
            return Ok(LookupReport::miss());
        }
        let blob_dir = self.blob_dir(context.compatibility_id);
        let candidates = self.discover_candidates(&blob_dir, request_tokens, context)?;
        let mut examined = 0usize;
        let mut corrupt_removed = 0usize;
        for candidate in candidates {
            examined += 1;
            let Some(lease) = self.open_candidate(&candidate.path)? else {
                continue;
            };
            if !admit(candidate.matched_len, lease.size) {
                continue;
            }
            match decode_snapshot(&mut &lease.file, context.codec_constraints()) {
                Ok(snapshot)
                    if namespace_matches(
                        &snapshot,
                        request_tokens,
                        candidate.matched_len,
                        candidate.mode,
                        context.compatibility_id,
                        &candidate.digest,
                    ) =>
                {
                    let touched = self.touch_if_same_inode(&candidate.path, &lease).is_ok();
                    return Ok(LookupReport {
                        snapshot: Some(snapshot),
                        matched_prefix_len: candidate.matched_len,
                        restored_prefix_len: candidate.mode.restored_len(candidate.matched_len),
                        exact: candidate.matched_len == request_tokens.len(),
                        candidates_examined: examined,
                        corrupt_entries_removed: corrupt_removed,
                        touched,
                    });
                }
                Ok(_) => {
                    if self.remove_if_same_inode(&candidate.path, &lease)? {
                        corrupt_removed += 1;
                    }
                }
                Err(error) if codec_error_proves_invalid_blob(&error) => {
                    if self.remove_if_same_inode(&candidate.path, &lease)? {
                        corrupt_removed += 1;
                    }
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(LookupReport {
            candidates_examined: examined,
            corrupt_entries_removed: corrupt_removed,
            ..LookupReport::miss()
        })
    }

    fn publish_staged(
        &self,
        context: StoreContext<'_>,
        snapshot: &SessionSnapshot,
        mode: SnapshotMode,
        digest: [u8; 32],
        final_path: &Path,
        staged: &StagedBlob,
        staging_cleanup: StagingCleanupReport,
    ) -> Result<PublishReport, CheckpointStoreError> {
        let mut repaired = false;
        for _ in 0..MAX_PUBLICATION_ATTEMPTS {
            if let Some(lease) = self.open_candidate(final_path)? {
                let valid = match decode_snapshot(&mut &lease.file, context.codec_constraints()) {
                    Ok(existing) => {
                        same_snapshot_prefix(&existing, snapshot)
                            && existing.matched_prefix_len() == snapshot.matched_prefix_len()
                            && SnapshotMode::from_snapshot(&existing) == mode
                            && snapshot_prefix_key(context.compatibility_id, &existing) == digest
                    }
                    Err(error) if codec_error_proves_invalid_blob(&error) => false,
                    Err(error) => return Err(error.into()),
                };
                if valid {
                    if let Some(report) =
                        self.admit_existing(final_path, &lease, staged.integrity, staging_cleanup)?
                    {
                        return Ok(report);
                    }
                    continue;
                }
                repaired |= self.remove_if_same_inode(final_path, &lease)?;
                continue;
            }

            let lock = self.namespace.lock_exclusive()?;
            if path_exists_nofollow(final_path)? {
                drop(lock);
                continue;
            }
            let before = scan_managed_blobs(&self.blobs_root(), is_managed_blob_name)?;
            let (evicted_entries, evicted_bytes, remaining_bytes) = evict_to_fit(
                before,
                staged.encoded.record_bytes,
                self.max_managed_blob_bytes,
                Some(final_path),
            )?;
            let source_meta = metadata_nofollow(&staged.path)?.ok_or(
                CheckpointStoreError::StagedMetadata("staging path disappeared"),
            )?;
            let source_stamp = FileStamp::from_metadata(&source_meta);
            validate_staged_stamp(
                &source_stamp,
                staged.encoded.record_bytes,
                1,
                Some(&staged.synced),
            )?;
            if let Err(source) = std::fs::hard_link(&staged.path, final_path) {
                if evicted_entries > 0 {
                    return Err(CheckpointStoreError::PostMutationIo {
                        operation: "publish blob after eviction",
                        source,
                    });
                }
                return Err(source.into());
            }
            let staged_meta = metadata_nofollow(&staged.path)
                .map_err(|source| CheckpointStoreError::PostMutationIo {
                    operation: "stat staged blob after publication",
                    source,
                })?
                .ok_or(CheckpointStoreError::PostCommit("staged blob disappeared"))?;
            let final_meta = metadata_nofollow(final_path)
                .map_err(|source| CheckpointStoreError::PostMutationIo {
                    operation: "stat final blob after publication",
                    source,
                })?
                .ok_or(CheckpointStoreError::PostCommit("final disappeared"))?;
            let fd_stamp = FileStamp::from_metadata(&staged.file.metadata().map_err(|source| {
                CheckpointStoreError::PostMutationIo {
                    operation: "stat staged descriptor after publication",
                    source,
                }
            })?);
            let staged_stamp = FileStamp::from_metadata(&staged_meta);
            let final_stamp = FileStamp::from_metadata(&final_meta);
            validate_post_link_stamp(&fd_stamp, staged.encoded.record_bytes, &staged.opening)
                .map_err(|_| CheckpointStoreError::PostCommit("staged descriptor drifted"))?;
            validate_post_link_stamp(&staged_stamp, staged.encoded.record_bytes, &fd_stamp)
                .map_err(|_| CheckpointStoreError::PostCommit("staged path drifted"))?;
            validate_post_link_stamp(&final_stamp, staged.encoded.record_bytes, &fd_stamp)
                .map_err(|_| CheckpointStoreError::PostCommit("final blob drifted"))?;
            sync_directory(final_path.parent().expect("blob parent")).map_err(|source| {
                CheckpointStoreError::PostMutationIo {
                    operation: "sync published blob directory",
                    source,
                }
            })?;
            drop(lock);
            return Ok(PublishReport {
                outcome: if repaired {
                    PublishOutcome::RepairedCorrupt
                } else {
                    PublishOutcome::Published
                },
                blob_bytes: staged.encoded.record_bytes,
                managed_bytes_after: remaining_bytes
                    .checked_add(staged.encoded.record_bytes)
                    .ok_or(CheckpointStoreError::ManagedBytesOverflow)?,
                evicted_entries,
                evicted_bytes,
                touched: false,
                staged_integrity: staged.integrity,
                staging_entries_removed: staging_cleanup.removed_entries,
                staging_allocated_bytes_reclaimed: staging_cleanup.reclaimed_bytes,
                staging_entries_examined: staging_cleanup.examined_entries,
                staging_live_entries: staging_cleanup.live_entries,
                staging_legacy_entries: staging_cleanup.legacy_entries,
                staging_foreign_entries: staging_cleanup.foreign_entries,
                staging_cleanup_truncated: staging_cleanup.truncated,
            });
        }
        Err(CheckpointStoreError::ConcurrentChurn)
    }

    fn encode_staged(
        &self,
        temp_path: &Path,
        context: StoreContext<'_>,
        snapshot: &SessionSnapshot,
    ) -> Result<StagedBlob, CheckpointStoreError> {
        let mut file = self.namespace.create_staging_file(
            temp_path,
            self.staged_integrity == StagedIntegrityMode::Decode,
        )?;
        let opening = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&opening, 0, 1, None)?;
        let encoded = {
            let mut writer = BufWriter::new(&mut file);
            let encoded = encode_snapshot(&mut writer, snapshot, context.codec_constraints())?;
            writer.flush()?;
            encoded
        };
        file.sync_all()?;
        let integrity_t0 = Instant::now();
        let synced = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&synced, encoded.record_bytes, 1, Some(&opening))?;
        if self.staged_integrity == StagedIntegrityMode::Decode {
            file.seek(SeekFrom::Start(0))?;
            let decoded = decode_snapshot(&mut file, context.codec_constraints())?;
            if !same_snapshot_prefix(&decoded, snapshot)
                || SnapshotMode::from_snapshot(&decoded) != SnapshotMode::from_snapshot(snapshot)
            {
                return Err(CheckpointStoreError::StagedValidation);
            }
        }
        Ok(StagedBlob {
            file,
            path: temp_path.to_path_buf(),
            encoded,
            opening,
            synced,
            integrity: StagedIntegrityReport {
                mode: self.staged_integrity,
                elapsed: integrity_t0.elapsed(),
            },
        })
    }

    fn discover_candidates(
        &self,
        blob_dir: &Path,
        request_tokens: &[i32],
        context: StoreContext<'_>,
    ) -> Result<Vec<Candidate>, CheckpointStoreError> {
        let lock = self.namespace.lock_shared()?;
        let mut found = Vec::new();
        let mut lengths = BTreeSet::new();
        if !require_real_directory_if_exists(&self.blobs_root())?
            || !require_real_directory_if_exists(blob_dir)?
        {
            drop(lock);
            return Ok(Vec::new());
        }
        let entries = match std::fs::read_dir(blob_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                drop(lock);
                return Ok(Vec::new());
            }
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let Some(parsed) = parse_blob_name(&entry.file_name()) else {
                continue;
            };
            if parsed.matched_len <= request_tokens.len() {
                lengths.insert(parsed.matched_len);
                found.push((entry.path(), parsed));
            }
        }
        drop(lock);
        let digests = request_prefix_keys(context.compatibility_id, request_tokens, &lengths);
        let mut candidates = Vec::new();
        for (path, parsed) in found {
            if parsed.matched_len == request_tokens.len() && parsed.mode == SnapshotMode::Consumed {
                continue;
            }
            if digests.get(&parsed.matched_len) == Some(&parsed.digest) {
                candidates.push(Candidate {
                    path,
                    matched_len: parsed.matched_len,
                    mode: parsed.mode,
                    digest: parsed.digest,
                });
            }
        }
        candidates.sort_by(|a, b| {
            b.matched_len
                .cmp(&a.matched_len)
                .then_with(|| {
                    mode_rank(a.mode, a.matched_len == request_tokens.len())
                        .cmp(&mode_rank(b.mode, b.matched_len == request_tokens.len()))
                })
                .then_with(|| a.path.cmp(&b.path))
        });
        Ok(candidates)
    }

    fn admit_existing(
        &self,
        path: &Path,
        lease: &BlobLease,
        staged_integrity: StagedIntegrityReport,
        staging_cleanup: StagingCleanupReport,
    ) -> Result<Option<PublishReport>, CheckpointStoreError> {
        let _lock = self.namespace.lock_exclusive()?;
        let Some(metadata) = metadata_nofollow(path)? else {
            return Ok(None);
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Ok(None);
        }
        if metadata.dev() != lease.dev || metadata.ino() != lease.ino {
            return Ok(None);
        }
        let before = scan_managed_blobs(&self.blobs_root(), is_managed_blob_name)?;
        let (evicted_entries, evicted_bytes, managed_bytes_after) =
            evict_to_fit(before, 0, self.max_managed_blob_bytes, Some(path))?;
        let touched = lease
            .file
            .set_times(FileTimes::new().set_modified(SystemTime::now()))
            .is_ok();
        Ok(Some(PublishReport {
            outcome: PublishOutcome::ExistingValid,
            blob_bytes: lease.size,
            managed_bytes_after,
            evicted_entries,
            evicted_bytes,
            touched,
            staged_integrity,
            staging_entries_removed: staging_cleanup.removed_entries,
            staging_allocated_bytes_reclaimed: staging_cleanup.reclaimed_bytes,
            staging_entries_examined: staging_cleanup.examined_entries,
            staging_live_entries: staging_cleanup.live_entries,
            staging_legacy_entries: staging_cleanup.legacy_entries,
            staging_foreign_entries: staging_cleanup.foreign_entries,
            staging_cleanup_truncated: staging_cleanup.truncated,
        }))
    }

    fn open_candidate(&self, path: &Path) -> Result<Option<BlobLease>, CheckpointStoreError> {
        Ok(self.namespace.open_candidate(path)?)
    }

    fn remove_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointStoreError> {
        Ok(self.namespace.remove_if_same_inode(path, lease)?)
    }

    fn touch_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<(), CheckpointStoreError> {
        Ok(self.namespace.touch_if_same_inode(path, lease)?)
    }

    fn ensure_blob_dir(
        &self,
        blob_dir: &Path,
    ) -> Result<StagingCleanupReport, CheckpointStoreError> {
        Ok(self.namespace.ensure_blob_dir(blob_dir)?)
    }

    #[cfg(test)]
    fn lock_exclusive(&self) -> Result<crate::checkpoint_fs::StoreLock, CheckpointStoreError> {
        Ok(self.namespace.lock_exclusive()?)
    }

    fn blobs_root(&self) -> PathBuf {
        self.namespace.blobs_root()
    }

    fn blob_dir(&self, compatibility_id: &[u8; 32]) -> PathBuf {
        self.namespace.blob_dir(compatibility_id)
    }
}

fn is_managed_blob_name(name: &std::ffi::OsStr) -> bool {
    parse_blob_name(name).is_some()
}

impl From<CheckpointFsError> for CheckpointStoreError {
    fn from(error: CheckpointFsError) -> Self {
        match error {
            CheckpointFsError::Io(source) => Self::Io(source),
            CheckpointFsError::ForeignEntryAtKey(path) => Self::ForeignEntryAtKey(path),
            CheckpointFsError::ManagedBytesOverflow => Self::ManagedBytesOverflow,
            CheckpointFsError::OversizedBlob {
                blob_bytes,
                max_managed_blob_bytes,
            } => Self::OversizedBlob {
                blob_bytes,
                max_managed_blob_bytes,
            },
            CheckpointFsError::PostMutationIo { operation, source } => {
                Self::PostMutationIo { operation, source }
            }
            CheckpointFsError::StagedMetadata(reason) => Self::StagedMetadata(reason),
            CheckpointFsError::TouchLostRace => Self::TouchLostRace,
        }
    }
}

#[derive(Clone, Copy)]
pub struct StoreContext<'a> {
    pub compatibility_id: &'a [u8; 32],
    pub identity: &'a SnapshotIdentity,
    pub vocab_size: usize,
    pub max_context_tokens: usize,
    pub max_record_bytes: u64,
}

impl StoreContext<'_> {
    fn codec_constraints(&self) -> SnapshotCodecConstraints<'_> {
        SnapshotCodecConstraints {
            expected_identity: self.identity,
            expected_compatibility_id: self.compatibility_id,
            expected_vocab_size: self.vocab_size,
            max_context_tokens: self.max_context_tokens,
            max_record_bytes: self.max_record_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishOutcome {
    Published,
    ExistingValid,
    RepairedCorrupt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StagedIntegrityMode {
    Decode,
    DeferredRestore,
}

impl StagedIntegrityMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "decode" => Some(Self::Decode),
            "deferred-restore" => Some(Self::DeferredRestore),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::DeferredRestore => "deferred-restore",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StagedIntegrityReport {
    pub mode: StagedIntegrityMode,
    pub elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublishReport {
    pub outcome: PublishOutcome,
    pub blob_bytes: u64,
    pub managed_bytes_after: u64,
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
    pub touched: bool,
    pub staged_integrity: StagedIntegrityReport,
    pub staging_entries_removed: usize,
    pub staging_allocated_bytes_reclaimed: u64,
    pub staging_entries_examined: usize,
    pub staging_live_entries: usize,
    pub staging_legacy_entries: usize,
    pub staging_foreign_entries: usize,
    pub staging_cleanup_truncated: bool,
}

#[derive(Debug)]
pub struct LookupReport {
    pub snapshot: Option<SessionSnapshot>,
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub candidates_examined: usize,
    pub corrupt_entries_removed: usize,
    pub touched: bool,
}

impl LookupReport {
    fn miss() -> Self {
        Self {
            snapshot: None,
            matched_prefix_len: 0,
            restored_prefix_len: 0,
            exact: false,
            candidates_examined: 0,
            corrupt_entries_removed: 0,
            touched: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CheckpointStoreError {
    #[error("checkpoint store I/O: {0}")]
    Io(#[from] io::Error),
    #[error("checkpoint store codec: {0}")]
    Codec(#[from] SnapshotCodecError),
    #[error("checkpoint blob size {blob_bytes} exceeds managed budget {max_managed_blob_bytes}")]
    OversizedBlob {
        blob_bytes: u64,
        max_managed_blob_bytes: u64,
    },
    #[error("foreign non-regular entry at managed checkpoint key: {0}")]
    ForeignEntryAtKey(PathBuf),
    #[error("checkpoint managed-byte accounting overflow")]
    ManagedBytesOverflow,
    #[error("checkpoint publication changed namespace but failed: {0}")]
    PostCommit(&'static str),
    #[error("checkpoint store mutated namespace before {operation} failed: {source}")]
    PostMutationIo {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("checkpoint staged validation failed")]
    StagedValidation,
    #[error("checkpoint staged metadata failed: {0}")]
    StagedMetadata(&'static str),
    #[error("checkpoint store changed repeatedly during publication")]
    ConcurrentChurn,
    #[error("checkpoint touch lost an eviction or replacement race")]
    TouchLostRace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum SnapshotMode {
    Consumed,
    ConsumedWithLogits,
    Pending,
    PendingWithLogits,
}

impl SnapshotMode {
    fn from_snapshot(snapshot: &SessionSnapshot) -> Self {
        match (
            snapshot.pending_token.is_some(),
            snapshot.final_logits.is_some(),
        ) {
            (false, false) => Self::Consumed,
            (false, true) => Self::ConsumedWithLogits,
            (true, false) => Self::Pending,
            (true, true) => Self::PendingWithLogits,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Consumed => "c0",
            Self::ConsumedWithLogits => "c1",
            Self::Pending => "p0",
            Self::PendingWithLogits => "p1",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "c0" => Some(Self::Consumed),
            "c1" => Some(Self::ConsumedWithLogits),
            "p0" => Some(Self::Pending),
            "p1" => Some(Self::PendingWithLogits),
            _ => None,
        }
    }

    fn restored_len(self, matched_len: usize) -> usize {
        if matches!(self, Self::Pending | Self::PendingWithLogits) {
            matched_len.saturating_sub(1)
        } else {
            matched_len
        }
    }
}

fn mode_rank(mode: SnapshotMode, exact: bool) -> usize {
    if exact {
        match mode {
            SnapshotMode::ConsumedWithLogits => 0,
            SnapshotMode::Pending => 1,
            SnapshotMode::PendingWithLogits => 2,
            SnapshotMode::Consumed => 3,
        }
    } else {
        match mode {
            SnapshotMode::ConsumedWithLogits => 0,
            SnapshotMode::Consumed => 1,
            SnapshotMode::Pending => 2,
            SnapshotMode::PendingWithLogits => 3,
        }
    }
}

#[derive(Clone)]
struct Candidate {
    path: PathBuf,
    matched_len: usize,
    mode: SnapshotMode,
    digest: [u8; 32],
}

struct StagedBlob {
    file: std::fs::File,
    path: PathBuf,
    encoded: EncodedSnapshot,
    opening: FileStamp,
    synced: FileStamp,
    integrity: StagedIntegrityReport,
}

struct ParsedBlobName {
    matched_len: usize,
    mode: SnapshotMode,
    digest: [u8; 32],
}

fn snapshot_prefix_key(compatibility_id: &[u8; 32], snapshot: &SessionSnapshot) -> [u8; 32] {
    let mut hasher = prefix_key_hasher(compatibility_id);
    for &token in &snapshot.prefix_tokens {
        hasher.update(&token.to_le_bytes());
    }
    if let Some(token) = snapshot.pending_token {
        hasher.update(&token.to_le_bytes());
    }
    hasher.update(&(snapshot.matched_prefix_len() as u64).to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn request_prefix_keys(
    compatibility_id: &[u8; 32],
    request_tokens: &[i32],
    lengths: &BTreeSet<usize>,
) -> HashMap<usize, [u8; 32]> {
    let mut out = HashMap::with_capacity(lengths.len());
    let mut hasher = prefix_key_hasher(compatibility_id);
    for (index, &token) in request_tokens.iter().enumerate() {
        hasher.update(&token.to_le_bytes());
        let length = index + 1;
        if lengths.contains(&length) {
            let mut at_length = hasher.clone();
            at_length.update(&(length as u64).to_le_bytes());
            out.insert(length, *at_length.finalize().as_bytes());
        }
    }
    out
}

fn prefix_key_hasher(compatibility_id: &[u8; 32]) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_KEY_DOMAIN);
    hasher.update(compatibility_id);
    hasher
}

fn namespace_matches(
    snapshot: &SessionSnapshot,
    request_tokens: &[i32],
    matched_len: usize,
    mode: SnapshotMode,
    compatibility_id: &[u8; 32],
    digest: &[u8; 32],
) -> bool {
    snapshot.matched_prefix_len() == matched_len
        && SnapshotMode::from_snapshot(snapshot) == mode
        && same_request_prefix(snapshot, request_tokens)
        && snapshot_prefix_key(compatibility_id, snapshot) == *digest
}

fn same_request_prefix(snapshot: &SessionSnapshot, request_tokens: &[i32]) -> bool {
    snapshot
        .prefix_tokens
        .iter()
        .copied()
        .chain(snapshot.pending_token)
        .eq(request_tokens
            .iter()
            .copied()
            .take(snapshot.matched_prefix_len()))
}

fn same_snapshot_prefix(a: &SessionSnapshot, b: &SessionSnapshot) -> bool {
    a.prefix_tokens == b.prefix_tokens && a.pending_token == b.pending_token
}

fn blob_name(matched_len: usize, mode: SnapshotMode, digest: &[u8; 32]) -> String {
    format!(
        "{matched_len}-{}-{}.{}",
        mode.label(),
        hex(digest),
        BLOB_EXTENSION
    )
}

fn parse_blob_name(name: &std::ffi::OsStr) -> Option<ParsedBlobName> {
    let name = name.to_str()?;
    let stem = name.strip_suffix(&format!(".{BLOB_EXTENSION}"))?;
    let mut parts = stem.split('-');
    let length_text = parts.next()?;
    let mode = SnapshotMode::parse(parts.next()?)?;
    let digest_text = parts.next()?;
    if parts.next().is_some()
        || length_text.is_empty()
        || (length_text.len() > 1 && length_text.starts_with('0'))
        || digest_text.len() != 64
    {
        return None;
    }
    let matched_len = length_text.parse::<usize>().ok()?;
    let digest = parse_hex_32(digest_text)?;
    Some(ParsedBlobName {
        matched_len,
        mode,
        digest,
    })
}

fn codec_error_proves_invalid_blob(error: &SnapshotCodecError) -> bool {
    match error {
        SnapshotCodecError::InvalidHeader(_)
        | SnapshotCodecError::CompatibilityMismatch
        | SnapshotCodecError::IdentityMismatch
        | SnapshotCodecError::ArithmeticOverflow { .. }
        | SnapshotCodecError::SectionLength { .. }
        | SnapshotCodecError::DigestMismatch
        | SnapshotCodecError::TrailingBytes => true,
        SnapshotCodecError::Snapshot(
            crate::metal_forward::SnapshotValidationError::AllocationFailed { .. },
        ) => false,
        SnapshotCodecError::Snapshot(_) => true,
        SnapshotCodecError::Io(error) => error.kind() == io::ErrorKind::UnexpectedEof,
        SnapshotCodecError::UnsupportedByteOrder
        | SnapshotCodecError::RecordBudgetExceeded { .. }
        | SnapshotCodecError::ContextCapacityExceeded { .. }
        | SnapshotCodecError::AllocationFailed { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_fs::{MAX_STAGING_SCAVENGE_PER_PUBLISH, TEMP_PREFIX};
    use crate::metal_forward::{
        SNAPSHOT_LAYOUT_VERSION, SnapshotKvStorageKind, SnapshotValidationError,
    };
    use sha2::{Digest, Sha256};
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom};
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    const COMPATIBILITY_ID: [u8; 32] = [0x3c; 32];
    const OTHER_COMPATIBILITY_ID: [u8; 32] = [0x5d; 32];

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qwen-checkpoint-store-{label}-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct ChildGuard(Option<std::process::Child>);

    impl ChildGuard {
        fn spawn(command: &mut std::process::Command) -> Self {
            Self(Some(
                command.spawn().expect("spawn checkpoint staging helper"),
            ))
        }

        fn child_mut(&mut self) -> &mut std::process::Child {
            self.0.as_mut().expect("checkpoint staging helper exists")
        }

        fn kill_and_wait(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                child.wait().expect("wait for checkpoint staging helper");
            }
        }
    }

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            self.kill_and_wait();
        }
    }

    fn identity() -> SnapshotIdentity {
        SnapshotIdentity {
            model_id: 1,
            tokenizer_id: 2,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 1,
            kv_dim_elements: 2,
            kv_bytes_per_token: 4,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 4,
            gdn_conv_elements_per_layer: 3,
        }
    }

    fn snapshot(tokens: &[i32], pending: Option<i32>, logits: bool) -> SessionSnapshot {
        let identity = identity();
        let prefix_len = tokens.len();
        let kv_bytes =
            identity.n_attn_layers as usize * prefix_len * identity.kv_bytes_per_token as usize;
        SessionSnapshot {
            identity,
            prefix_tokens: tokens.to_vec(),
            pending_token: pending,
            kv_n_pos: vec![prefix_len; 2],
            kv_k_arena: vec![0x11; kv_bytes],
            kv_v_arena: vec![0x22; kv_bytes],
            gdn_conv_arena: vec![0x33; 12],
            gdn_state_arena: vec![0x44; 16],
            final_logits: logits.then(|| vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            capture_tail: None,
        }
    }

    fn context<'a>(identity: &'a SnapshotIdentity) -> StoreContext<'a> {
        StoreContext {
            compatibility_id: &COMPATIBILITY_ID,
            identity,
            vocab_size: 5,
            max_context_tokens: 8,
            max_record_bytes: 1 << 20,
        }
    }

    fn blob_path(store: &DurableCheckpointStore, snapshot: &SessionSnapshot) -> PathBuf {
        blob_path_for(store, &COMPATIBILITY_ID, snapshot)
    }

    fn blob_path_for(
        store: &DurableCheckpointStore,
        compatibility_id: &[u8; 32],
        snapshot: &SessionSnapshot,
    ) -> PathBuf {
        let digest = snapshot_prefix_key(compatibility_id, snapshot);
        store.blob_dir(compatibility_id).join(blob_name(
            snapshot.matched_prefix_len(),
            SnapshotMode::from_snapshot(snapshot),
            &digest,
        ))
    }

    fn assert_snapshot_equal(a: &SessionSnapshot, b: &SessionSnapshot) {
        assert_eq!(a.identity.abi(), b.identity.abi());
        assert_eq!(a.prefix_tokens, b.prefix_tokens);
        assert_eq!(a.pending_token, b.pending_token);
        assert_eq!(a.kv_n_pos, b.kv_n_pos);
        assert_eq!(a.kv_k_arena, b.kv_k_arena);
        assert_eq!(a.kv_v_arena, b.kv_v_arena);
        assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena);
        assert_eq!(a.gdn_state_arena, b.gdn_state_arena);
        assert_eq!(
            a.final_logits.as_ref().map(|values| values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()),
            b.final_logits.as_ref().map(|values| values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>())
        );
    }

    fn sha256_path(path: &Path) -> String {
        let mut file = File::open(path).unwrap();
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        hex(&hasher.finalize())
    }

    #[test]
    fn staged_integrity_configuration_is_explicit_and_decode_default() {
        let temp = TestDir::new("integrity-config");
        let default = DurableCheckpointStore::new(&temp.0, 1 << 20);
        assert_eq!(default.staged_integrity_mode(), StagedIntegrityMode::Decode);
        assert!(!default.staged_integrity_is_explicit());

        let deferred = DurableCheckpointStore::with_staged_integrity(
            &temp.0,
            1 << 20,
            StagedIntegrityMode::DeferredRestore,
        );
        assert_eq!(
            deferred.staged_integrity_mode(),
            StagedIntegrityMode::DeferredRestore
        );
        assert!(deferred.staged_integrity_is_explicit());
        assert_eq!(
            StagedIntegrityMode::parse("decode"),
            Some(StagedIntegrityMode::Decode)
        );
        assert_eq!(
            StagedIntegrityMode::parse("deferred-restore"),
            Some(StagedIntegrityMode::DeferredRestore)
        );
        for invalid in ["", "Decode", "deferred_restore", "deferred", "true"] {
            assert_eq!(StagedIntegrityMode::parse(invalid), None);
        }
    }

    #[test]
    fn staged_file_stamp_validator_rejects_every_contract_mismatch() {
        let opening = FileStamp {
            regular: true,
            len: 0,
            mode: 0o600,
            dev: 7,
            ino: 11,
            nlink: 1,
        };
        let synced = FileStamp {
            len: 123,
            ..opening
        };
        validate_staged_stamp(&opening, 0, 1, None).unwrap();
        validate_staged_stamp(&synced, 123, 1, Some(&opening)).unwrap();

        for invalid in [
            FileStamp {
                regular: false,
                ..synced
            },
            FileStamp { len: 122, ..synced },
            FileStamp {
                mode: 0o640,
                ..synced
            },
            FileStamp { dev: 8, ..synced },
            FileStamp { ino: 12, ..synced },
            FileStamp { nlink: 2, ..synced },
        ] {
            assert!(validate_staged_stamp(&invalid, 123, 1, Some(&opening)).is_err());
        }
        let linked = FileStamp { nlink: 2, ..synced };
        validate_post_link_stamp(&linked, 123, &opening).unwrap();
    }

    #[test]
    fn staged_integrity_modes_emit_identical_small_records() {
        let temp = TestDir::new("integrity-records");
        let snapshot = snapshot(&[1, 2], Some(3), true);
        let decode_path = temp.0.join("decode.qcp");
        let deferred_path = temp.0.join("deferred.qcp");
        let decode_store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let deferred_store = DurableCheckpointStore::with_staged_integrity(
            &temp.0,
            1 << 20,
            StagedIntegrityMode::DeferredRestore,
        );

        let decode = decode_store
            .encode_staged(&decode_path, context(&snapshot.identity), &snapshot)
            .unwrap();
        let deferred = deferred_store
            .encode_staged(&deferred_path, context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(decode.encoded, deferred.encoded);
        assert_eq!(decode.integrity.mode, StagedIntegrityMode::Decode);
        assert_eq!(
            deferred.integrity.mode,
            StagedIntegrityMode::DeferredRestore
        );
        let deferred_flags = unsafe { libc::fcntl(deferred.file.as_raw_fd(), libc::F_GETFL) };
        assert!(deferred_flags >= 0);
        assert_eq!(deferred_flags & libc::O_ACCMODE, libc::O_WRONLY);
        assert_eq!(
            std::fs::read(&decode_path).unwrap(),
            std::fs::read(&deferred_path).unwrap()
        );

        let final_path = temp.0.join("final.qcp");
        std::fs::hard_link(&deferred.path, &final_path).unwrap();
        let fd_meta = deferred.file.metadata().unwrap();
        let staged_meta = deferred.path.metadata().unwrap();
        let final_meta = final_path.metadata().unwrap();
        assert_eq!(
            (fd_meta.dev(), fd_meta.ino()),
            (staged_meta.dev(), staged_meta.ino())
        );
        assert_eq!(
            (fd_meta.dev(), fd_meta.ino()),
            (final_meta.dev(), final_meta.ino())
        );
        assert_eq!(fd_meta.nlink(), 2);
        assert_eq!(staged_meta.nlink(), 2);
        assert_eq!(final_meta.nlink(), 2);
        std::fs::remove_file(&deferred.path).unwrap();
        assert_eq!(final_path.metadata().unwrap().nlink(), 1);

        let restored = decode_snapshot(
            &mut File::open(&final_path).unwrap(),
            context(&snapshot.identity).codec_constraints(),
        )
        .unwrap();
        assert_snapshot_equal(&snapshot, &restored);
    }

    #[test]
    fn publication_reclaims_only_abandoned_canonical_staging_files() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TestDir::new("staging-scavenge");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first_snapshot = snapshot(&[1, 2], None, true);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();

        let abandoned = unique_temp_path(&blob_dir, &[0x11; 32]);
        std::fs::write(&abandoned, b"abandoned").unwrap();
        std::fs::set_permissions(&abandoned, std::fs::Permissions::from_mode(0o600)).unwrap();

        let live = unique_temp_path(&blob_dir, &[0x22; 32]);
        let live_file = store.namespace.create_staging_file(&live, true).unwrap();
        (&live_file).write_all(b"live-staging-data").unwrap();

        let foreign = blob_dir.join(format!("{TEMP_PREFIX}foreign-lookalike"));
        std::fs::write(&foreign, b"foreign").unwrap();
        let wrong_mode = unique_temp_path(&blob_dir, &[0x33; 32]);
        std::fs::write(&wrong_mode, b"wrong-mode").unwrap();
        std::fs::set_permissions(&wrong_mode, std::fs::Permissions::from_mode(0o640)).unwrap();

        let report = store
            .publish(context(&first_snapshot.identity), &first_snapshot)
            .unwrap();
        assert_eq!(report.staging_entries_removed, 1);
        assert!(report.staging_allocated_bytes_reclaimed >= 9);
        assert_eq!(report.staging_live_entries, 1);
        assert_eq!(report.staging_foreign_entries, 2);
        assert!(!abandoned.exists());
        assert!(live.exists());
        assert!(foreign.exists());
        assert!(wrong_mode.exists());

        drop(live_file);
        let next = snapshot(&[1, 3], None, true);
        let report = store.publish(context(&next.identity), &next).unwrap();
        assert_eq!(report.staging_entries_removed, 1);
        assert!(report.staging_allocated_bytes_reclaimed >= 17);
        assert_eq!(report.staging_live_entries, 0);
        assert_eq!(report.staging_foreign_entries, 2);
        assert!(!live.exists());
        assert!(foreign.exists());
        assert!(wrong_mode.exists());
    }

    #[test]
    fn managed_blob_probe_ignores_staging_and_foreign_entries() {
        let temp = TestDir::new("managed-probe");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();
        std::fs::write(unique_temp_path(&blob_dir, &[0x44; 32]), b"temp").unwrap();
        std::fs::write(blob_dir.join("foreign.qcp"), b"foreign").unwrap();
        assert!(!store.has_managed_blobs().unwrap());

        let first_snapshot = snapshot(&[1, 2], None, true);
        store
            .publish(context(&first_snapshot.identity), &first_snapshot)
            .unwrap();
        assert!(store.has_managed_blobs().unwrap());
    }

    #[test]
    fn managed_blob_probe_still_rejects_substituted_compatibility_directory() {
        let temp = TestDir::new("managed-probe-foreign-compat");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        let outside = temp.0.join("outside-probe-compat");
        std::fs::create_dir(&outside).unwrap();
        let substituted = store.blob_dir(&OTHER_COMPATIBILITY_ID);
        std::os::unix::fs::symlink(&outside, &substituted).unwrap();

        assert!(matches!(
            store.has_managed_blobs(),
            Err(CheckpointStoreError::ForeignEntryAtKey(path)) if path == substituted
        ));
        assert!(blob_path(&store, &snapshot).exists());
    }

    #[test]
    fn legacy_unlocked_staging_file_is_never_reclaimed_automatically() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TestDir::new("legacy-staging-grace");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first_snapshot = snapshot(&[1, 2], None, true);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();
        let legacy = blob_dir.join(format!(
            "{TEMP_PREFIX}{}-1-0-{}",
            std::process::id(),
            "55".repeat(8)
        ));
        std::fs::write(&legacy, b"legacy").unwrap();
        std::fs::set_permissions(&legacy, std::fs::Permissions::from_mode(0o600)).unwrap();

        let report = store
            .publish(context(&first_snapshot.identity), &first_snapshot)
            .unwrap();
        assert_eq!(report.staging_entries_removed, 0);
        assert_eq!(report.staging_live_entries, 0);
        assert_eq!(report.staging_legacy_entries, 1);
        assert!(legacy.exists());

        let next = snapshot(&[1, 3], None, true);
        let report = store.publish(context(&next.identity), &next).unwrap();
        assert_eq!(report.staging_entries_removed, 0);
        assert_eq!(report.staging_allocated_bytes_reclaimed, 0);
        assert_eq!(report.staging_legacy_entries, 1);
        assert!(legacy.exists());
    }

    #[test]
    fn scavenging_post_link_staging_name_does_not_overstate_reclaimed_bytes() {
        let temp = TestDir::new("linked-staging-scavenge");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        store.publish(context(&first.identity), &first).unwrap();
        let final_path = blob_path(&store, &first);
        let blob_dir = final_path.parent().unwrap();
        let stale_link = unique_temp_path(blob_dir, &[0x66; 32]);
        std::fs::hard_link(&final_path, &stale_link).unwrap();
        assert_eq!(final_path.metadata().unwrap().nlink(), 2);

        let next = snapshot(&[1, 3], None, true);
        let report = store.publish(context(&next.identity), &next).unwrap();
        assert_eq!(report.staging_entries_removed, 1);
        assert_eq!(report.staging_allocated_bytes_reclaimed, 0);
        assert!(!stale_link.exists());
        assert_eq!(final_path.metadata().unwrap().nlink(), 1);
    }

    #[test]
    fn scavenging_two_staging_aliases_counts_physical_reclaim_once() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TestDir::new("aliased-staging-scavenge");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();
        let first_link = unique_temp_path(&blob_dir, &[0x77; 32]);
        let second_link = unique_temp_path(&blob_dir, &[0x88; 32]);
        std::fs::write(&first_link, b"aliased-staging").unwrap();
        std::fs::set_permissions(&first_link, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::hard_link(&first_link, &second_link).unwrap();
        assert_eq!(first_link.metadata().unwrap().nlink(), 2);

        let snapshot = snapshot(&[1, 2], None, true);
        let report = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(report.staging_entries_removed, 2);
        assert!(report.staging_allocated_bytes_reclaimed >= 15);
        assert!(!first_link.exists());
        assert!(!second_link.exists());
    }

    #[test]
    fn staging_cleanup_is_bounded_and_reports_truncation() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TestDir::new("bounded-staging-scavenge");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();
        let staging_count = MAX_STAGING_SCAVENGE_PER_PUBLISH + 1;
        for index in 0..staging_count {
            let mut digest = [0u8; 32];
            digest[..8].copy_from_slice(&(index as u64).to_le_bytes());
            let path = unique_temp_path(&blob_dir, &digest);
            std::fs::write(&path, b"x").unwrap();
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }

        let snapshot = snapshot(&[1, 2], None, true);
        let report = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(
            report.staging_entries_examined,
            MAX_STAGING_SCAVENGE_PER_PUBLISH
        );
        assert_eq!(
            report.staging_entries_removed,
            MAX_STAGING_SCAVENGE_PER_PUBLISH
        );
        assert!(
            report.staging_allocated_bytes_reclaimed >= MAX_STAGING_SCAVENGE_PER_PUBLISH as u64
        );
        assert!(report.staging_cleanup_truncated);
    }

    #[test]
    #[ignore = "spawned by cross-process staging-lock test"]
    fn staging_lock_child_helper() {
        let Some(root) = std::env::var_os("QWEN_CHECKPOINT_STAGING_CHILD_ROOT") else {
            return;
        };
        let ready = PathBuf::from(
            std::env::var_os("QWEN_CHECKPOINT_STAGING_CHILD_READY")
                .expect("staging helper ready path"),
        );
        let store = DurableCheckpointStore::new(PathBuf::from(root), 1 << 20);
        let blob_dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&blob_dir).unwrap();
        let path = unique_temp_path(&blob_dir, &[0x99; 32]);
        let file = store.namespace.create_staging_file(&path, true).unwrap();
        (&file).write_all(&[0x5a; 23]).unwrap();
        std::fs::write(&ready, path.to_string_lossy().as_bytes()).unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    #[test]
    fn cross_process_staging_lock_survives_cleanup_and_releases_on_death() {
        let temp = TestDir::new("cross-process-staging-lock");
        let ready = temp.0.join("child-ready");
        let executable = std::env::current_exe().unwrap();
        let mut command = std::process::Command::new(executable);
        command
            .arg("--exact")
            .arg("checkpoint_store::tests::staging_lock_child_helper")
            .arg("--ignored")
            .env("QWEN_CHECKPOINT_STAGING_CHILD_ROOT", &temp.0)
            .env("QWEN_CHECKPOINT_STAGING_CHILD_READY", &ready)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = ChildGuard::spawn(&mut command);
        for _ in 0..500 {
            if ready.exists() {
                break;
            }
            assert!(
                child.child_mut().try_wait().unwrap().is_none(),
                "checkpoint staging helper exited before ready"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            ready.exists(),
            "checkpoint staging helper did not become ready"
        );
        let staging_path = PathBuf::from(std::fs::read_to_string(&ready).unwrap());

        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        let report = store.publish(context(&first.identity), &first).unwrap();
        assert_eq!(report.staging_entries_removed, 0);
        assert_eq!(report.staging_live_entries, 1);
        assert!(staging_path.exists());

        child.kill_and_wait();
        let next = snapshot(&[1, 3], None, true);
        let report = store.publish(context(&next.identity), &next).unwrap();
        assert_eq!(report.staging_entries_removed, 1);
        assert!(report.staging_allocated_bytes_reclaimed >= 23);
        assert!(!staging_path.exists());
    }

    #[test]
    fn store_publishes_and_finds_exact_or_extended_prefix() {
        let temp = TestDir::new("basic");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        assert!(!store.has_managed_blobs().unwrap());
        let snapshot = snapshot(&[1, 2], None, true);
        let published = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(published.outcome, PublishOutcome::Published);
        assert!(store.has_managed_blobs().unwrap());
        let existing = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(existing.outcome, PublishOutcome::ExistingValid);

        let exact = store.lookup(context(&snapshot.identity), &[1, 2]).unwrap();
        assert!(exact.exact);
        assert_eq!(exact.matched_prefix_len, 2);
        assert_eq!(exact.restored_prefix_len, 2);
        assert_snapshot_equal(&snapshot, exact.snapshot.as_ref().unwrap());

        let extension = store
            .lookup(context(&snapshot.identity), &[1, 2, 4])
            .unwrap();
        assert!(!extension.exact);
        assert_eq!(extension.matched_prefix_len, 2);
    }

    #[test]
    fn store_ranks_pending_and_consumed_modes_by_request_shape() {
        let temp = TestDir::new("modes");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let pending = snapshot(&[1, 2], Some(3), false);
        let consumed = snapshot(&[1, 2, 3], None, false);
        store.publish(context(&pending.identity), &pending).unwrap();
        store
            .publish(context(&consumed.identity), &consumed)
            .unwrap();

        let exact = store.lookup(context(&identity()), &[1, 2, 3]).unwrap();
        assert_eq!(exact.restored_prefix_len, 2);
        assert_eq!(exact.snapshot.unwrap().pending_token, Some(3));

        let extension = store.lookup(context(&identity()), &[1, 2, 3, 4]).unwrap();
        assert_eq!(extension.restored_prefix_len, 3);
        assert_eq!(extension.snapshot.unwrap().pending_token, None);

        let consumed_logits = snapshot(&[1, 2, 3], None, true);
        store
            .publish(context(&consumed_logits.identity), &consumed_logits)
            .unwrap();
        let exact = store.lookup(context(&identity()), &[1, 2, 3]).unwrap();
        assert_eq!(exact.restored_prefix_len, 3);
        assert!(exact.snapshot.unwrap().final_logits.is_some());
    }

    #[test]
    fn store_keeps_first_valid_writer_for_same_mode() {
        let temp = TestDir::new("first-writer");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        let mut second = first.clone();
        second.kv_k_arena[0] ^= 0xff;
        store.publish(context(&first.identity), &first).unwrap();
        let report = store.publish(context(&second.identity), &second).unwrap();
        assert_eq!(report.outcome, PublishOutcome::ExistingValid);
        let restored = store
            .lookup(context(&first.identity), &[1, 2, 3])
            .unwrap()
            .snapshot
            .unwrap();
        assert_eq!(restored.kv_k_arena, first.kv_k_arena);
    }

    #[test]
    fn store_repairs_corrupt_winner_conditionally() {
        let temp = TestDir::new("repair");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        let path = blob_path(&store, &snapshot);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        file.write_all(&[byte[0] ^ 1]).unwrap();
        file.sync_all().unwrap();

        let report = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(report.outcome, PublishOutcome::RepairedCorrupt);
        let restored = store.lookup(context(&snapshot.identity), &[1, 2]).unwrap();
        assert!(restored.snapshot.is_some());
    }

    #[test]
    fn deferred_restore_rejects_corruption_on_first_lookup() {
        let temp = TestDir::new("deferred-corruption");
        let store = DurableCheckpointStore::with_staged_integrity(
            &temp.0,
            1 << 20,
            StagedIntegrityMode::DeferredRestore,
        );
        let snapshot = snapshot(&[1, 2], None, true);
        let published = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        assert_eq!(published.outcome, PublishOutcome::Published);
        assert_eq!(
            published.staged_integrity.mode,
            StagedIntegrityMode::DeferredRestore
        );

        let path = blob_path(&store, &snapshot);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        file.write_all(&[byte[0] ^ 1]).unwrap();
        file.sync_all().unwrap();

        let lookup = store.lookup(context(&snapshot.identity), &[1, 2]).unwrap();
        assert!(lookup.snapshot.is_none());
        assert_eq!(lookup.candidates_examined, 1);
        assert_eq!(lookup.corrupt_entries_removed, 1);
        assert!(!path.exists());
    }

    #[test]
    fn store_falls_back_after_removing_corrupt_longest_match() {
        let temp = TestDir::new("corrupt-fallback");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let shorter = snapshot(&[1], None, true);
        let longer = snapshot(&[1, 2], None, true);
        store.publish(context(&shorter.identity), &shorter).unwrap();
        store.publish(context(&longer.identity), &longer).unwrap();

        let longer_path = blob_path(&store, &longer);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&longer_path)
            .unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        file.write_all(&[byte[0] ^ 1]).unwrap();
        file.sync_all().unwrap();

        let report = store.lookup(context(&longer.identity), &[1, 2, 3]).unwrap();
        assert_eq!(report.matched_prefix_len, 1);
        assert_eq!(report.candidates_examined, 2);
        assert_eq!(report.corrupt_entries_removed, 1);
        assert_snapshot_equal(&shorter, report.snapshot.as_ref().unwrap());
        assert!(!longer_path.exists());
    }

    #[test]
    fn store_falls_back_to_same_length_mode_after_corruption() {
        let temp = TestDir::new("corrupt-mode-fallback");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let consumed = snapshot(&[1, 2], None, true);
        let pending = snapshot(&[1], Some(2), false);
        store
            .publish(context(&consumed.identity), &consumed)
            .unwrap();
        store.publish(context(&pending.identity), &pending).unwrap();

        let consumed_path = blob_path(&store, &consumed);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&consumed_path)
            .unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        file.seek(SeekFrom::Start(16 * 1024)).unwrap();
        file.write_all(&[byte[0] ^ 1]).unwrap();
        file.sync_all().unwrap();

        let report = store.lookup(context(&consumed.identity), &[1, 2]).unwrap();
        assert_eq!(report.matched_prefix_len, 2);
        assert_eq!(report.restored_prefix_len, 1);
        assert_eq!(report.candidates_examined, 2);
        assert_eq!(report.corrupt_entries_removed, 1);
        assert_snapshot_equal(&pending, report.snapshot.as_ref().unwrap());
    }

    #[test]
    fn store_keeps_valid_blob_on_caller_budget_failure() {
        let temp = TestDir::new("caller-budget");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        let published = store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        let path = blob_path(&store, &snapshot);
        let mut constrained = context(&snapshot.identity);
        constrained.max_record_bytes = published.blob_bytes - 1;

        assert!(matches!(
            store.lookup(constrained, &[1, 2]),
            Err(CheckpointStoreError::Codec(
                SnapshotCodecError::RecordBudgetExceeded { .. }
            ))
        ));
        assert!(path.exists());
        assert!(matches!(
            store.publish(constrained, &snapshot),
            Err(CheckpointStoreError::Codec(
                SnapshotCodecError::RecordBudgetExceeded { .. }
            ))
        ));
        assert!(path.exists());
        assert!(
            store
                .lookup(context(&snapshot.identity), &[1, 2])
                .unwrap()
                .snapshot
                .is_some()
        );
    }

    #[test]
    fn store_enforces_global_logical_budget_and_open_lease() {
        let temp = TestDir::new("budget");
        let generous = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        let first_report = generous.publish(context(&first.identity), &first).unwrap();
        let first_path = blob_path(&generous, &first);
        let lease = generous.open_candidate(&first_path).unwrap().unwrap();

        let constrained = DurableCheckpointStore::new(&temp.0, first_report.blob_bytes);
        let second = snapshot(&[1, 3], None, true);
        let second_report = constrained
            .publish(context(&second.identity), &second)
            .unwrap();
        assert_eq!(second_report.evicted_entries, 1);
        assert!(!first_path.exists());

        let restored = decode_snapshot(
            &mut &lease.file,
            context(&first.identity).codec_constraints(),
        )
        .expect("open lease survives unlink");
        assert_snapshot_equal(&first, &restored);
        assert!(
            constrained
                .lookup(context(&first.identity), &[1, 2])
                .unwrap()
                .snapshot
                .is_none()
        );
        assert!(
            constrained
                .lookup(context(&second.identity), &[1, 3])
                .unwrap()
                .snapshot
                .is_some()
        );

        let too_small = DurableCheckpointStore::new(&temp.0, first_report.blob_bytes - 1);
        let third = snapshot(&[2, 3], None, true);
        assert!(matches!(
            too_small.publish(context(&third.identity), &third),
            Err(CheckpointStoreError::OversizedBlob { .. })
        ));
    }

    #[test]
    fn store_enforces_tighter_budget_on_existing_valid_publish() {
        let temp = TestDir::new("existing-budget");
        let generous = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        let second = snapshot(&[2, 3], None, true);
        let first_report = generous.publish(context(&first.identity), &first).unwrap();
        let second_context = StoreContext {
            compatibility_id: &OTHER_COMPATIBILITY_ID,
            ..context(&second.identity)
        };
        generous.publish(second_context, &second).unwrap();
        let second_path = blob_path_for(&generous, &OTHER_COMPATIBILITY_ID, &second);

        let constrained = DurableCheckpointStore::new(&temp.0, first_report.blob_bytes);
        let report = constrained
            .publish(context(&first.identity), &first)
            .unwrap();
        assert_eq!(report.outcome, PublishOutcome::ExistingValid);
        assert_eq!(report.evicted_entries, 1);
        assert!(report.managed_bytes_after <= first_report.blob_bytes);
        assert!(!second_path.exists());
        assert_eq!(
            scan_managed_blobs(&constrained.blobs_root(), is_managed_blob_name)
                .unwrap()
                .total_bytes,
            report.managed_bytes_after
        );
    }

    #[test]
    fn store_evicts_oldest_managed_blob_by_mtime() {
        use std::time::Duration;

        let temp = TestDir::new("mtime-lru");
        let generous = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let first = snapshot(&[1, 2], None, true);
        let second = snapshot(&[2, 3], None, true);
        let third = snapshot(&[3, 4], None, true);
        let first_report = generous.publish(context(&first.identity), &first).unwrap();
        generous
            .publish(context(&second.identity), &second)
            .unwrap();
        let first_path = blob_path(&generous, &first);
        let second_path = blob_path(&generous, &second);
        File::open(&first_path)
            .unwrap()
            .set_times(
                FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1)),
            )
            .unwrap();
        File::open(&second_path)
            .unwrap()
            .set_times(
                FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(2)),
            )
            .unwrap();

        let constrained = DurableCheckpointStore::new(&temp.0, first_report.blob_bytes * 2);
        let report = constrained
            .publish(context(&third.identity), &third)
            .unwrap();
        assert_eq!(report.evicted_entries, 1);
        assert!(!first_path.exists());
        assert!(second_path.exists());
        assert!(blob_path(&constrained, &third).exists());
    }

    #[test]
    fn mixed_integrity_publishers_converge_on_one_valid_inode() {
        let temp = TestDir::new("concurrent");
        let decode = Arc::new(DurableCheckpointStore::new(&temp.0, 1 << 20));
        let deferred = Arc::new(DurableCheckpointStore::with_staged_integrity(
            &temp.0,
            1 << 20,
            StagedIntegrityMode::DeferredRestore,
        ));
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();
        for index in 0..8 {
            let store = if index % 2 == 0 {
                Arc::clone(&decode)
            } else {
                Arc::clone(&deferred)
            };
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                let snapshot = snapshot(&[1, 2], None, true);
                barrier.wait();
                store
                    .publish(context(&snapshot.identity), &snapshot)
                    .unwrap()
                    .outcome
            }));
        }
        let outcomes: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(
            outcomes
                .iter()
                .filter(|&&outcome| outcome == PublishOutcome::Published)
                .count(),
            1
        );
        let restored = decode.lookup(context(&identity()), &[1, 2]).unwrap();
        assert!(restored.snapshot.is_some());
        assert_eq!(
            scan_managed_blobs(&decode.blobs_root(), is_managed_blob_name)
                .unwrap()
                .blobs
                .len(),
            1
        );
    }

    #[test]
    fn store_rejects_foreign_entry_at_exact_key() {
        let temp = TestDir::new("foreign");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        let dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&dir).unwrap();
        let target = temp.0.join("foreign-target");
        std::fs::write(&target, b"foreign").unwrap();
        std::os::unix::fs::symlink(target, blob_path(&store, &snapshot)).unwrap();
        assert!(matches!(
            store.publish(context(&snapshot.identity), &snapshot),
            Err(CheckpointStoreError::ForeignEntryAtKey(_))
        ));
    }

    #[test]
    fn store_rejects_nonregular_entry_at_exact_key_without_blocking() {
        use std::os::unix::ffi::OsStrExt;

        let temp = TestDir::new("foreign-fifo");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        let dir = store.blob_dir(&COMPATIBILITY_ID);
        store.ensure_blob_dir(&dir).unwrap();
        let path = blob_path(&store, &snapshot);
        let path_bytes = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path_bytes.as_ptr(), 0o600) }, 0);

        assert!(matches!(
            store.publish(context(&snapshot.identity), &snapshot),
            Err(CheckpointStoreError::ForeignEntryAtKey(found)) if found == path
        ));
    }

    #[test]
    fn store_rejects_substituted_managed_directories() {
        let snapshot = snapshot(&[1, 2], None, true);

        let root_temp = TestDir::new("foreign-blobs-root");
        let root_store = DurableCheckpointStore::new(&root_temp.0, 1 << 20);
        drop(root_store.lock_exclusive().unwrap());
        let outside_root = root_temp.0.join("outside-root");
        let outside_compat = outside_root.join(hex(&COMPATIBILITY_ID));
        std::fs::create_dir_all(&outside_compat).unwrap();
        let outside_blob =
            outside_compat.join(blob_path(&root_store, &snapshot).file_name().unwrap());
        std::fs::write(&outside_blob, b"outside").unwrap();
        std::os::unix::fs::symlink(&outside_root, root_store.blobs_root()).unwrap();
        assert!(matches!(
            root_store.lookup(context(&snapshot.identity), &[1, 2]),
            Err(CheckpointStoreError::ForeignEntryAtKey(path))
                if path == root_store.blobs_root()
        ));
        assert!(outside_blob.exists());

        let compat_temp = TestDir::new("foreign-compat-dir");
        let compat_store = DurableCheckpointStore::new(&compat_temp.0, 1 << 20);
        drop(compat_store.lock_exclusive().unwrap());
        std::fs::create_dir(compat_store.blobs_root()).unwrap();
        let outside_compat = compat_temp.0.join("outside-compat");
        std::fs::create_dir(&outside_compat).unwrap();
        let outside_blob =
            outside_compat.join(blob_path(&compat_store, &snapshot).file_name().unwrap());
        std::fs::write(&outside_blob, b"outside").unwrap();
        std::os::unix::fs::symlink(&outside_compat, compat_store.blob_dir(&COMPATIBILITY_ID))
            .unwrap();
        assert!(matches!(
            compat_store.lookup(context(&snapshot.identity), &[1, 2]),
            Err(CheckpointStoreError::ForeignEntryAtKey(path))
                if path == compat_store.blob_dir(&COMPATIBILITY_ID)
        ));
        assert!(outside_blob.exists());
    }

    #[test]
    fn store_rejects_substituted_foreign_compatibility_during_global_scan() {
        let temp = TestDir::new("foreign-global-compat");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], None, true);
        store
            .publish(context(&snapshot.identity), &snapshot)
            .unwrap();
        let outside = temp.0.join("outside-global-compat");
        std::fs::create_dir(&outside).unwrap();
        let substituted = store.blob_dir(&OTHER_COMPATIBILITY_ID);
        std::os::unix::fs::symlink(&outside, &substituted).unwrap();

        assert!(matches!(
            store.publish(context(&snapshot.identity), &snapshot),
            Err(CheckpointStoreError::ForeignEntryAtKey(path)) if path == substituted
        ));
        assert!(outside.exists());
        assert!(blob_path(&store, &snapshot).exists());
    }

    #[test]
    fn store_corruption_classifier_preserves_resource_and_transient_failures() {
        for error in [
            SnapshotCodecError::RecordBudgetExceeded {
                record_bytes: 2,
                max_record_bytes: 1,
            },
            SnapshotCodecError::ContextCapacityExceeded {
                prefix_len: 2,
                max_context_tokens: 1,
            },
            SnapshotCodecError::AllocationFailed {
                section: "test",
                bytes: 1,
            },
            SnapshotCodecError::Snapshot(SnapshotValidationError::AllocationFailed {
                section: "test",
                bytes: 1,
            }),
            SnapshotCodecError::Io(io::Error::other("transient")),
        ] {
            assert!(!codec_error_proves_invalid_blob(&error));
        }
        assert!(codec_error_proves_invalid_blob(
            &SnapshotCodecError::DigestMismatch
        ));
        assert!(codec_error_proves_invalid_blob(&SnapshotCodecError::Io(
            io::Error::new(io::ErrorKind::UnexpectedEof, "truncated")
        )));
    }

    #[test]
    #[ignore = "v0.636 exact-size filesystem floor"]
    fn checkpoint_deferred_restore_exact_size_floor() {
        const EXPECTED_BYTES: u64 = 582_854_188;
        const EXPECTED_SHA256: &str =
            "69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65";
        const EXPECTED_ENCODER_BLAKE3: &str =
            "2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67";
        const EXPECTED_RELATIVE: &str = concat!(
            "v1/blobs/",
            "fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e/",
            "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
        );
        const STORE_BUDGET: u64 = 805_306_368;
        const MAX_CONTEXT: usize = 6_516;
        const VOCAB_SIZE: usize = 248_320;

        let fixture = PathBuf::from(
            std::env::var("QWEN_CHECKPOINT_FLOOR_FIXTURE").expect("QWEN_CHECKPOINT_FLOOR_FIXTURE"),
        );
        let root = PathBuf::from(
            std::env::var("QWEN_CHECKPOINT_FLOOR_ROOT").expect("QWEN_CHECKPOINT_FLOOR_ROOT"),
        );
        let mode_value = std::env::var("QWEN_CHECKPOINT_STAGED_INTEGRITY")
            .expect("QWEN_CHECKPOINT_STAGED_INTEGRITY");
        let mode = StagedIntegrityMode::parse(&mode_value).expect("floor integrity mode");
        assert!(root.is_dir());
        assert!(root.read_dir().unwrap().next().is_none());
        assert_eq!(fixture.metadata().unwrap().len(), EXPECTED_BYTES);
        assert_eq!(sha256_path(&fixture), EXPECTED_SHA256);

        let identity = SnapshotIdentity {
            model_id: 7_081_852_628_295_403_893,
            tokenizer_id: 11_867_181_210_256_840_983,
            layout_version: 4,
            n_attn_layers: 16,
            n_gdn_layers: 48,
            kv_dim_elements: 1_024,
            kv_bytes_per_token: 2_048,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 786_432,
            gdn_conv_elements_per_layer: 30_720,
        };
        let compatibility =
            parse_hex_32("fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e")
                .unwrap();
        let context = StoreContext {
            compatibility_id: &compatibility,
            identity: &identity,
            vocab_size: VOCAB_SIZE,
            max_context_tokens: MAX_CONTEXT,
            max_record_bytes: STORE_BUDGET,
        };
        let snapshot = decode_snapshot(
            &mut File::open(&fixture).unwrap(),
            context.codec_constraints(),
        )
        .unwrap();
        assert_eq!(snapshot.prefix_len(), 6_499);
        assert_eq!(snapshot.matched_prefix_len(), 6_500);
        assert_eq!(snapshot.pending_token, Some(248_068));

        let store = DurableCheckpointStore::with_staged_integrity(&root, STORE_BUDGET, mode);
        let publish_t0 = Instant::now();
        let report = store.publish(context, &snapshot).unwrap();
        let publish_us = publish_t0.elapsed().as_micros();
        assert_eq!(report.outcome, PublishOutcome::Published);
        assert_eq!(report.evicted_entries, 0);
        assert_eq!(report.managed_bytes_after, EXPECTED_BYTES);
        assert_eq!(report.blob_bytes, EXPECTED_BYTES);
        assert_eq!(report.staged_integrity.mode, mode);

        let blob = blob_path_for(&store, &compatibility, &snapshot);
        assert_eq!(
            blob.strip_prefix(&root).unwrap(),
            Path::new(EXPECTED_RELATIVE)
        );
        let metadata = blob.metadata().unwrap();
        assert_eq!(metadata.len(), EXPECTED_BYTES);
        assert_eq!(metadata.mode() & 0o7777, 0o600);
        assert_eq!(metadata.nlink(), 1);
        let temp_files = blob
            .parent()
            .unwrap()
            .read_dir()
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with(TEMP_PREFIX))
            .count();
        assert_eq!(temp_files, 0);

        let mut trailer_file = File::open(&blob).unwrap();
        trailer_file.seek(SeekFrom::End(-32)).unwrap();
        let mut encoder_digest = [0u8; 32];
        trailer_file.read_exact(&mut encoder_digest).unwrap();
        assert_eq!(hex(&encoder_digest), EXPECTED_ENCODER_BLAKE3);
        assert_eq!(sha256_path(&blob), EXPECTED_SHA256);

        let decode_t0 = Instant::now();
        let decoded =
            decode_snapshot(&mut File::open(&blob).unwrap(), context.codec_constraints()).unwrap();
        let full_decode_us = decode_t0.elapsed().as_micros();
        assert_snapshot_equal(&snapshot, &decoded);

        let outcome = match report.outcome {
            PublishOutcome::Published => "published",
            PublishOutcome::ExistingValid => "existing_valid",
            PublishOutcome::RepairedCorrupt => "repaired_corrupt",
        };
        println!(
            concat!(
                "[checkpoint-deferred-floor] schema=1 mode={} record_bytes={} ",
                "encoder_blake3={} blob_sha256={} staged_integrity_us={} ",
                "publish_us={} full_decode_us={} outcome={} evicted={} ",
                "managed_bytes_after={} vocab_size={} max_context={} ",
                "max_record_bytes={} store_budget_bytes={} matched={} restored={} ",
                "pending={} file_mode={:04o} nlink={} temp_files={} blob_relative={}"
            ),
            mode.as_str(),
            report.blob_bytes,
            hex(&encoder_digest),
            EXPECTED_SHA256,
            report.staged_integrity.elapsed.as_micros(),
            publish_us,
            full_decode_us,
            outcome,
            report.evicted_entries,
            report.managed_bytes_after,
            VOCAB_SIZE,
            MAX_CONTEXT,
            context.max_record_bytes,
            STORE_BUDGET,
            snapshot.matched_prefix_len(),
            snapshot.prefix_len(),
            snapshot.pending_token.is_some(),
            metadata.mode() & 0o7777,
            metadata.nlink(),
            temp_files,
            blob.strip_prefix(&root).unwrap().display(),
        );
    }

    #[test]
    fn store_blob_name_parser_is_canonical() {
        let digest = [0xab; 32];
        let valid = blob_name(12, SnapshotMode::Pending, &digest);
        let parsed = parse_blob_name(std::ffi::OsStr::new(&valid)).unwrap();
        assert_eq!(parsed.matched_len, 12);
        assert_eq!(parsed.mode, SnapshotMode::Pending);
        assert_eq!(parsed.digest, digest);
        for invalid in [
            "012-p0-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.qcp",
            "12-p2-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.qcp",
            "12-p0-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.qcp",
            "12-p0-aa.qcp",
            "12-p0-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp",
        ] {
            assert!(parse_blob_name(std::ffi::OsStr::new(invalid)).is_none());
        }
    }

    #[test]
    fn store_rejects_semantically_invalid_snapshot_before_publication() {
        let temp = TestDir::new("invalid");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let mut snapshot = snapshot(&[1, 2], None, true);
        snapshot.prefix_tokens[0] = -1;
        assert!(matches!(
            store.publish(context(&snapshot.identity), &snapshot),
            Err(CheckpointStoreError::Codec(SnapshotCodecError::Snapshot(
                SnapshotValidationError::TokenOutOfRange { .. }
            )))
        ));
        assert_eq!(
            scan_managed_blobs(&store.blobs_root(), is_managed_blob_name)
                .unwrap()
                .blobs
                .len(),
            0
        );
    }
}
