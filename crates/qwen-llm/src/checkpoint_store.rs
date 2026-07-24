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
    EncodedSnapshot, SnapshotCodecConstraints, SnapshotCodecError, decode_snapshot,
    encode_snapshot, verify_staged_encoder_output,
};
use crate::checkpoint_identity::CheckpointIdentityCache;
use crate::metal_forward::{SessionSnapshot, SnapshotIdentity};
use std::collections::{BTreeSet, HashMap};
use std::fs::{File, FileTimes, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

const NAMESPACE_VERSION: &str = "v1";
const PREFIX_KEY_DOMAIN: &[u8] = b"qwen-checkpoint-prefix-key-v1\0";
const BLOB_EXTENSION: &str = "qcp";
const TEMP_PREFIX: &str = ".tmp-";
const LOCK_FILE: &str = "store.lock";
const MAX_PUBLICATION_ATTEMPTS: usize = 8;
const STAGED_VALIDATION_ENV: &str = "QWEN_CHECKPOINT_STAGED_VALIDATION";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct DurableCheckpointStore {
    root: PathBuf,
    max_managed_blob_bytes: u64,
    namespace_ready: Arc<AtomicBool>,
}

impl DurableCheckpointStore {
    pub fn new(root: impl Into<PathBuf>, max_managed_blob_bytes: u64) -> Self {
        Self {
            root: root.into(),
            max_managed_blob_bytes,
            namespace_ready: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn max_managed_blob_bytes(&self) -> u64 {
        self.max_managed_blob_bytes
    }

    pub fn identity_cache(&self) -> CheckpointIdentityCache {
        CheckpointIdentityCache::new(self.namespace_root().join("identity"))
    }

    /// Cheap global emptiness probe for callers that can skip strong identity
    /// resolution when no checkpoint blob could possibly match.
    pub fn has_managed_blobs(&self) -> Result<bool, CheckpointStoreError> {
        let _lock = self.lock_shared()?;
        Ok(!scan_managed_blobs(&self.blobs_root())?.blobs.is_empty())
    }

    pub fn publish(
        &self,
        context: StoreContext<'_>,
        snapshot: &SessionSnapshot,
    ) -> Result<PublishReport, CheckpointStoreError> {
        let staged_validation_mode = staged_validation_mode()?;
        let mode = SnapshotMode::from_snapshot(snapshot);
        let matched_len = snapshot.matched_prefix_len();
        let digest = snapshot_prefix_key(context.compatibility_id, snapshot);
        let blob_dir = self.blob_dir(context.compatibility_id);
        self.ensure_blob_dir(&blob_dir)?;
        let final_path = blob_dir.join(blob_name(matched_len, mode, &digest));
        let temp_path = unique_temp_path(&blob_dir, &digest);
        let (staged, staged_validation) = match self.encode_staged_with_mode(
            &temp_path,
            context,
            snapshot,
            staged_validation_mode,
        ) {
            Ok(staged) => staged,
            Err(error) => {
                let _ = std::fs::remove_file(&temp_path);
                return Err(error);
            }
        };
        if staged.record_bytes > self.max_managed_blob_bytes {
            let _ = std::fs::remove_file(&temp_path);
            return Err(CheckpointStoreError::OversizedBlob {
                blob_bytes: staged.record_bytes,
                max_managed_blob_bytes: self.max_managed_blob_bytes,
            });
        }

        let result = self.publish_staged(
            context,
            snapshot,
            mode,
            digest,
            &temp_path,
            &final_path,
            staged,
            staged_validation,
        );
        let _ = std::fs::remove_file(&temp_path);
        result
    }

    pub fn lookup(
        &self,
        context: StoreContext<'_>,
        request_tokens: &[i32],
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
        temp_path: &Path,
        final_path: &Path,
        staged: EncodedSnapshot,
        staged_validation: StagedValidationReport,
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
                        self.admit_existing(final_path, &lease, staged_validation)?
                    {
                        return Ok(report);
                    }
                    continue;
                }
                repaired |= self.remove_if_same_inode(final_path, &lease)?;
                continue;
            }

            let lock = self.lock_exclusive()?;
            if path_exists_nofollow(final_path)? {
                drop(lock);
                continue;
            }
            let before = scan_managed_blobs(&self.blobs_root())?;
            let (evicted_entries, evicted_bytes, remaining_bytes) = evict_to_fit(
                before,
                staged.record_bytes,
                self.max_managed_blob_bytes,
                Some(final_path),
            )?;
            if let Err(source) = std::fs::hard_link(temp_path, final_path) {
                if evicted_entries > 0 {
                    return Err(CheckpointStoreError::PostMutationIo {
                        operation: "publish blob after eviction",
                        source,
                    });
                }
                return Err(source.into());
            }
            let staged_meta = std::fs::metadata(temp_path).map_err(|source| {
                CheckpointStoreError::PostMutationIo {
                    operation: "stat staged blob after publication",
                    source,
                }
            })?;
            let final_meta = metadata_nofollow(final_path)
                .map_err(|source| CheckpointStoreError::PostMutationIo {
                    operation: "stat final blob after publication",
                    source,
                })?
                .ok_or_else(|| CheckpointStoreError::PostCommit("final disappeared"))?;
            if !same_inode(&staged_meta, &final_meta) {
                return Err(CheckpointStoreError::PostCommit("final inode mismatch"));
            }
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
                blob_bytes: staged.record_bytes,
                managed_bytes_after: remaining_bytes
                    .checked_add(staged.record_bytes)
                    .ok_or(CheckpointStoreError::ManagedBytesOverflow)?,
                evicted_entries,
                evicted_bytes,
                touched: false,
                staged_validation,
            });
        }
        Err(CheckpointStoreError::ConcurrentChurn)
    }

    fn encode_staged_with_mode(
        &self,
        temp_path: &Path,
        context: StoreContext<'_>,
        snapshot: &SessionSnapshot,
        mode: StagedValidationMode,
    ) -> Result<(EncodedSnapshot, StagedValidationReport), CheckpointStoreError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW);
        let mut file = options.open(temp_path)?;
        let encoded = {
            let mut writer = BufWriter::new(&mut file);
            let encoded = encode_snapshot(&mut writer, snapshot, context.codec_constraints())?;
            writer.flush()?;
            encoded
        };
        file.sync_all()?;
        file.seek(SeekFrom::Start(0))?;
        let validation_started = Instant::now();
        match mode {
            StagedValidationMode::MaterializedDecode => {
                let decoded = decode_snapshot(&mut file, context.codec_constraints())?;
                if !same_snapshot_prefix(&decoded, snapshot)
                    || SnapshotMode::from_snapshot(&decoded)
                        != SnapshotMode::from_snapshot(snapshot)
                {
                    return Err(CheckpointStoreError::StagedValidation);
                }
            }
            StagedValidationMode::EncoderDigest => {
                verify_staged_encoder_output(&mut file, encoded)?;
            }
        }
        Ok((
            encoded,
            StagedValidationReport {
                mode,
                elapsed: validation_started.elapsed(),
            },
        ))
    }

    fn discover_candidates(
        &self,
        blob_dir: &Path,
        request_tokens: &[i32],
        context: StoreContext<'_>,
    ) -> Result<Vec<Candidate>, CheckpointStoreError> {
        let lock = self.lock_shared()?;
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
            if parsed.matched_len == request_tokens.len()
                && parsed.mode == SnapshotMode::ConsumedNoLogits
            {
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

    fn ensure_blob_dir(&self, blob_dir: &Path) -> Result<(), CheckpointStoreError> {
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

    fn admit_existing(
        &self,
        path: &Path,
        lease: &BlobLease,
        staged_validation: StagedValidationReport,
    ) -> Result<Option<PublishReport>, CheckpointStoreError> {
        let _lock = self.lock_exclusive()?;
        let Some(metadata) = metadata_nofollow(path)? else {
            return Ok(None);
        };
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Ok(None);
        }
        if metadata.dev() != lease.dev || metadata.ino() != lease.ino {
            return Ok(None);
        }
        let before = scan_managed_blobs(&self.blobs_root())?;
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
            staged_validation,
        }))
    }

    fn open_candidate(&self, path: &Path) -> Result<Option<BlobLease>, CheckpointStoreError> {
        let _lock = self.lock_shared()?;
        open_blob_nofollow(path)
    }

    fn remove_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, CheckpointStoreError> {
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
            CheckpointStoreError::PostMutationIo {
                operation: "sync repaired blob directory",
                source,
            }
        })?;
        Ok(true)
    }

    fn touch_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<(), CheckpointStoreError> {
        let _lock = self.lock_exclusive()?;
        let metadata = metadata_nofollow(path)?.ok_or(CheckpointStoreError::TouchLostRace)?;
        if metadata.dev() != lease.dev || metadata.ino() != lease.ino {
            return Err(CheckpointStoreError::TouchLostRace);
        }
        lease
            .file
            .set_times(FileTimes::new().set_modified(SystemTime::now()))?;
        Ok(())
    }

    fn lock_shared(&self) -> Result<StoreLock, CheckpointStoreError> {
        self.lock(libc::LOCK_SH)
    }

    fn lock_exclusive(&self) -> Result<StoreLock, CheckpointStoreError> {
        self.lock(libc::LOCK_EX)
    }

    fn lock(&self, operation: libc::c_int) -> Result<StoreLock, CheckpointStoreError> {
        let initialize = !self.namespace_ready.load(Ordering::Acquire);
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
            return Err(CheckpointStoreError::ForeignEntryAtKey(
                self.namespace_root().join(LOCK_FILE),
            ));
        }
        if initialize {
            sync_directory(&namespace)?;
            self.namespace_ready.store(true, Ordering::Release);
        }
        flock_retry(&file, operation)?;
        Ok(StoreLock { file })
    }

    fn ensure_namespace(&self) -> Result<(), CheckpointStoreError> {
        create_directory_tree_synced(&self.root)?;
        let namespace = self.namespace_root();
        let namespace_created = create_directory(&namespace)?;
        ensure_real_directory(&namespace)?;
        if namespace_created {
            sync_directory(&self.root)?;
        }
        Ok(())
    }

    fn namespace_root(&self) -> PathBuf {
        self.root.join(NAMESPACE_VERSION)
    }

    fn blobs_root(&self) -> PathBuf {
        self.namespace_root().join("blobs")
    }

    fn blob_dir(&self, compatibility_id: &[u8; 32]) -> PathBuf {
        self.blobs_root().join(hex(compatibility_id))
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
pub enum StagedValidationMode {
    MaterializedDecode,
    EncoderDigest,
}

impl StagedValidationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaterializedDecode => "decode",
            Self::EncoderDigest => "encoder-digest",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StagedValidationReport {
    pub mode: StagedValidationMode,
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
    pub staged_validation: StagedValidationReport,
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
    #[error("invalid {STAGED_VALIDATION_ENV}; expected 'decode' or 'encoder-digest'")]
    InvalidStagedValidationMode,
    #[error("checkpoint store changed repeatedly during publication")]
    ConcurrentChurn,
    #[error("checkpoint touch lost an eviction or replacement race")]
    TouchLostRace,
}

fn staged_validation_mode() -> Result<StagedValidationMode, CheckpointStoreError> {
    match std::env::var(STAGED_VALIDATION_ENV) {
        Err(std::env::VarError::NotPresent) => parse_staged_validation_mode(None),
        Ok(value) => parse_staged_validation_mode(Some(&value)),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(CheckpointStoreError::InvalidStagedValidationMode)
        }
    }
}

fn parse_staged_validation_mode(
    value: Option<&str>,
) -> Result<StagedValidationMode, CheckpointStoreError> {
    match value {
        None | Some("decode") => Ok(StagedValidationMode::MaterializedDecode),
        Some("encoder-digest") => Ok(StagedValidationMode::EncoderDigest),
        Some(_) => Err(CheckpointStoreError::InvalidStagedValidationMode),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum SnapshotMode {
    ConsumedNoLogits,
    ConsumedLogits,
    PendingNoLogits,
    PendingLogits,
}

impl SnapshotMode {
    fn from_snapshot(snapshot: &SessionSnapshot) -> Self {
        match (
            snapshot.pending_token.is_some(),
            snapshot.final_logits.is_some(),
        ) {
            (false, false) => Self::ConsumedNoLogits,
            (false, true) => Self::ConsumedLogits,
            (true, false) => Self::PendingNoLogits,
            (true, true) => Self::PendingLogits,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ConsumedNoLogits => "c0",
            Self::ConsumedLogits => "c1",
            Self::PendingNoLogits => "p0",
            Self::PendingLogits => "p1",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "c0" => Some(Self::ConsumedNoLogits),
            "c1" => Some(Self::ConsumedLogits),
            "p0" => Some(Self::PendingNoLogits),
            "p1" => Some(Self::PendingLogits),
            _ => None,
        }
    }

    fn restored_len(self, matched_len: usize) -> usize {
        if matches!(self, Self::PendingNoLogits | Self::PendingLogits) {
            matched_len.saturating_sub(1)
        } else {
            matched_len
        }
    }
}

fn mode_rank(mode: SnapshotMode, exact: bool) -> usize {
    if exact {
        match mode {
            SnapshotMode::ConsumedLogits => 0,
            SnapshotMode::PendingNoLogits => 1,
            SnapshotMode::PendingLogits => 2,
            SnapshotMode::ConsumedNoLogits => 3,
        }
    } else {
        match mode {
            SnapshotMode::ConsumedLogits => 0,
            SnapshotMode::ConsumedNoLogits => 1,
            SnapshotMode::PendingNoLogits => 2,
            SnapshotMode::PendingLogits => 3,
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

struct BlobLease {
    file: File,
    dev: u64,
    ino: u64,
    size: u64,
}

struct StoreLock {
    file: File,
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = flock_retry(&self.file, libc::LOCK_UN);
    }
}

struct ParsedBlobName {
    matched_len: usize,
    mode: SnapshotMode,
    digest: [u8; 32],
}

struct ManagedBlob {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

struct ManagedScan {
    blobs: Vec<ManagedBlob>,
    total_bytes: u64,
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

fn unique_temp_path(blob_dir: &Path, digest: &[u8; 32]) -> PathBuf {
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

fn open_blob_nofollow(path: &Path) -> Result<Option<BlobLease>, CheckpointStoreError> {
    if let Some(metadata) = metadata_nofollow(path)?
        && (metadata.file_type().is_symlink() || !metadata.file_type().is_file())
    {
        return Err(CheckpointStoreError::ForeignEntryAtKey(path.to_path_buf()));
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
            return Err(CheckpointStoreError::ForeignEntryAtKey(path.to_path_buf()));
        }
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(CheckpointStoreError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(Some(BlobLease {
        file,
        dev: metadata.dev(),
        ino: metadata.ino(),
        size: metadata.len(),
    }))
}

fn metadata_nofollow(path: &Path) -> io::Result<Option<std::fs::Metadata>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn path_exists_nofollow(path: &Path) -> io::Result<bool> {
    Ok(metadata_nofollow(path)?.is_some())
}

fn ensure_real_directory(path: &Path) -> Result<(), CheckpointStoreError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CheckpointStoreError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(())
}

fn require_real_directory_if_exists(path: &Path) -> Result<bool, CheckpointStoreError> {
    let Some(metadata) = metadata_nofollow(path)? else {
        return Ok(false);
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CheckpointStoreError::ForeignEntryAtKey(path.to_path_buf()));
    }
    Ok(true)
}

fn create_directory(path: &Path) -> io::Result<bool> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

fn create_directory_tree_synced(path: &Path) -> Result<(), CheckpointStoreError> {
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

fn scan_managed_blobs(root: &Path) -> Result<ManagedScan, CheckpointStoreError> {
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
            return Err(CheckpointStoreError::ForeignEntryAtKey(compat_dir.path()));
        }
        for entry in std::fs::read_dir(compat_dir.path())? {
            let entry = entry?;
            if parse_blob_name(&entry.file_name()).is_none() {
                continue;
            }
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                continue;
            }
            total_bytes = total_bytes
                .checked_add(metadata.len())
                .ok_or(CheckpointStoreError::ManagedBytesOverflow)?;
            blobs.push(ManagedBlob {
                path: entry.path(),
                size: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }
    Ok(ManagedScan { blobs, total_bytes })
}

fn evict_to_fit(
    mut scan: ManagedScan,
    incoming_bytes: u64,
    max_bytes: u64,
    protected: Option<&Path>,
) -> Result<(usize, u64, u64), CheckpointStoreError> {
    scan.blobs.sort_by(|a, b| {
        a.modified
            .cmp(&b.modified)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut evicted_entries = 0usize;
    let mut evicted_bytes = 0u64;
    let mut affected_directories = BTreeSet::new();
    for blob in scan.blobs {
        if scan
            .total_bytes
            .checked_add(incoming_bytes)
            .ok_or(CheckpointStoreError::ManagedBytesOverflow)?
            <= max_bytes
        {
            break;
        }
        if protected.is_some_and(|path| path == blob.path) {
            continue;
        }
        if let Err(source) = std::fs::remove_file(&blob.path) {
            if evicted_entries > 0 {
                return Err(CheckpointStoreError::PostMutationIo {
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
            .ok_or(CheckpointStoreError::ManagedBytesOverflow)?;
    }
    if scan
        .total_bytes
        .checked_add(incoming_bytes)
        .ok_or(CheckpointStoreError::ManagedBytesOverflow)?
        > max_bytes
    {
        return Err(CheckpointStoreError::OversizedBlob {
            blob_bytes: incoming_bytes,
            max_managed_blob_bytes: max_bytes,
        });
    }
    for directory in affected_directories {
        sync_directory(&directory).map_err(|source| CheckpointStoreError::PostMutationIo {
            operation: "sync evicted blob directory",
            source,
        })?;
    }
    Ok((evicted_entries, evicted_bytes, scan.total_bytes))
}

fn flock_retry(file: &File, operation: libc::c_int) -> io::Result<()> {
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

fn same_inode(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.dev() == b.dev() && a.ino() == b.ino()
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn is_lower_hex_64(value: &std::ffi::OsStr) -> bool {
    value.to_str().is_some_and(|value| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn parse_hex_32(value: &str) -> Option<[u8; 32]> {
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

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_forward::{
        SNAPSHOT_LAYOUT_VERSION, SnapshotKvStorageKind, SnapshotValidationError,
    };
    use std::io::{Read, Seek, SeekFrom};
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

    #[test]
    fn staged_validation_mode_is_strict_and_defaults_to_decode() {
        assert_eq!(
            parse_staged_validation_mode(None).unwrap(),
            StagedValidationMode::MaterializedDecode
        );
        assert_eq!(
            parse_staged_validation_mode(Some("decode")).unwrap(),
            StagedValidationMode::MaterializedDecode
        );
        assert_eq!(
            parse_staged_validation_mode(Some("encoder-digest")).unwrap(),
            StagedValidationMode::EncoderDigest
        );
        for value in ["", "digest", "Decode", "encoder_digest", "ture"] {
            assert!(matches!(
                parse_staged_validation_mode(Some(value)),
                Err(CheckpointStoreError::InvalidStagedValidationMode)
            ));
        }
    }

    #[test]
    fn staged_validation_modes_emit_identical_records() {
        let temp = TestDir::new("staged-validation-modes");
        let store = DurableCheckpointStore::new(&temp.0, 1 << 20);
        let snapshot = snapshot(&[1, 2], Some(3), true);
        let decode_path = temp.0.join("decode.qcp");
        let digest_path = temp.0.join("encoder-digest.qcp");

        let (decode, decode_validation) = store
            .encode_staged_with_mode(
                &decode_path,
                context(&snapshot.identity),
                &snapshot,
                StagedValidationMode::MaterializedDecode,
            )
            .unwrap();
        let (digest, digest_validation) = store
            .encode_staged_with_mode(
                &digest_path,
                context(&snapshot.identity),
                &snapshot,
                StagedValidationMode::EncoderDigest,
            )
            .unwrap();

        assert_eq!(decode, digest);
        assert_eq!(
            decode_validation.mode,
            StagedValidationMode::MaterializedDecode
        );
        assert_eq!(digest_validation.mode, StagedValidationMode::EncoderDigest);
        assert_eq!(
            std::fs::read(&decode_path).unwrap(),
            std::fs::read(&digest_path).unwrap()
        );
        let restored = decode_snapshot(
            &mut File::open(&digest_path).unwrap(),
            context(&snapshot.identity).codec_constraints(),
        )
        .unwrap();
        assert_snapshot_equal(&snapshot, &restored);
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
            scan_managed_blobs(&constrained.blobs_root())
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
    fn store_concurrent_publishers_converge_on_one_valid_inode() {
        let temp = TestDir::new("concurrent");
        let store = Arc::new(DurableCheckpointStore::new(&temp.0, 1 << 20));
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
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
        let restored = store.lookup(context(&identity()), &[1, 2]).unwrap();
        assert!(restored.snapshot.is_some());
        assert_eq!(
            scan_managed_blobs(&store.blobs_root()).unwrap().blobs.len(),
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
            SnapshotCodecError::Io(io::Error::new(io::ErrorKind::Other, "transient")),
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
    fn store_blob_name_parser_is_canonical() {
        let digest = [0xab; 32];
        let valid = blob_name(12, SnapshotMode::PendingNoLogits, &digest);
        let parsed = parse_blob_name(std::ffi::OsStr::new(&valid)).unwrap();
        assert_eq!(parsed.matched_len, 12);
        assert_eq!(parsed.mode, SnapshotMode::PendingNoLogits);
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
            scan_managed_blobs(&store.blobs_root()).unwrap().blobs.len(),
            0
        );
    }
}
