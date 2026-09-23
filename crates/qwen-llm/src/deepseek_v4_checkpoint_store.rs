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

use crate::checkpoint_fs::{
    BlobLease, CheckpointFsError, FileStamp, StagingCleanupReport, StoreNamespace, evict_to_fit,
    has_managed_blob, hex, metadata_nofollow, parse_hex_32, path_exists_nofollow,
    require_real_directory_if_exists, scan_managed_blobs, sync_directory, unique_temp_path,
    validate_post_link_stamp, validate_staged_stamp,
};
use crate::checkpoint_identity::CheckpointIdentityCache;
use crate::checkpoint_store::{PublishOutcome, StagedIntegrityMode, StagedIntegrityReport};
use crate::deepseek_v4::DeepSeekV4Config;
use crate::deepseek_v4_metal::{
    DeepSeekV4CausalSnapshot, DeepSeekV4CompatibilityDigest, DeepSeekV4EncodedSnapshot,
    DeepSeekV4MetalError, DeepSeekV4ModelContentId, DeepSeekV4Session, DeepSeekV4SessionCapacity,
    DeepSeekV4SnapshotCodecConstraints, DeepSeekV4SnapshotCodecError, decode_causal_snapshot,
    encode_causal_snapshot,
};
use std::collections::{BTreeSet, HashMap};
use std::fs::FileTimes;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

const NAMESPACE_VERSION: &str = "dsv4-v1";
const PREFIX_KEY_DOMAIN: &[u8] = b"qwen-dsv4-checkpoint-prefix-key-v1\0";
const BLOB_EXTENSION: &str = "dsv4cp";
const MAX_PUBLICATION_ATTEMPTS: usize = 8;

#[derive(Clone, Debug)]
pub struct DeepSeekV4CheckpointStore {
    namespace: StoreNamespace,
    max_managed_blob_bytes: u64,
    staged_integrity: StagedIntegrityMode,
    staged_integrity_explicit: bool,
}

impl DeepSeekV4CheckpointStore {
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
    pub fn has_managed_blobs(&self) -> Result<bool, DeepSeekV4CheckpointStoreError> {
        let _lock = self.namespace.lock_shared()?;
        Ok(has_managed_blob(&self.blobs_root(), is_managed_blob_name)?)
    }

    pub fn publish(
        &self,
        context: DeepSeekV4StoreContext<'_>,
        snapshot: &DeepSeekV4CausalSnapshot,
    ) -> Result<DeepSeekV4PublishReport, DeepSeekV4CheckpointStoreError> {
        context.validate()?;
        let matched_len = snapshot.next_position() as usize;
        if snapshot.compatibility_digest() != context.compatibility_digest {
            return Err(DeepSeekV4CheckpointStoreError::CompatibilityMismatch);
        }
        let digest = snapshot_prefix_key(context.compatibility_digest.as_bytes(), snapshot);
        let blob_dir = self.blob_dir(context.compatibility_digest.as_bytes());
        let staging_cleanup = self.ensure_blob_dir(&blob_dir)?;
        let final_path = blob_dir.join(blob_name(matched_len, &digest));
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
            return Err(DeepSeekV4CheckpointStoreError::OversizedBlob {
                blob_bytes: staged.encoded.record_bytes,
                max_managed_blob_bytes: self.max_managed_blob_bytes,
            });
        }

        let result = self.publish_staged(
            context,
            snapshot,
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
        context: DeepSeekV4StoreContext<'_>,
        request_tokens: &[u32],
    ) -> Result<DeepSeekV4LookupReport, DeepSeekV4CheckpointStoreError> {
        self.lookup_filtered(context, request_tokens, |_, _| true)
    }

    /// [`Self::lookup`], consulting `admit(matched_len, blob_bytes)` before
    /// decoding each candidate (longest first); rejected candidates are
    /// skipped unread. The decoded snapshot is returned by value, so callers
    /// can share it (e.g. promote it into a RAM cache) before restoring.
    pub fn lookup_filtered(
        &self,
        context: DeepSeekV4StoreContext<'_>,
        request_tokens: &[u32],
        mut admit: impl FnMut(usize, u64) -> bool,
    ) -> Result<DeepSeekV4LookupReport, DeepSeekV4CheckpointStoreError> {
        context.validate()?;
        if request_tokens.is_empty() {
            return Ok(DeepSeekV4LookupReport::miss());
        }
        let blob_dir = self.blob_dir(context.compatibility_digest.as_bytes());
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
            match decode_causal_snapshot(&mut &lease.file, context.codec_constraints()) {
                Ok(snapshot)
                    if namespace_matches(
                        &snapshot,
                        request_tokens,
                        candidate.matched_len,
                        context.compatibility_digest.as_bytes(),
                        &candidate.digest,
                    ) =>
                {
                    let touched = self.touch_if_same_inode(&candidate.path, &lease).is_ok();
                    return Ok(DeepSeekV4LookupReport {
                        snapshot: Some(snapshot),
                        matched_prefix_len: candidate.matched_len,
                        restored_prefix_len: candidate.matched_len,
                        exact: false,
                        candidates_examined: examined,
                        corrupt_entries_removed: corrupt_removed,
                        touched,
                    });
                }
                Ok(snapshot)
                    if snapshot.next_position() as usize == candidate.matched_len
                        && same_request_prefix(&snapshot, request_tokens)
                        && snapshot.compatibility_digest() == context.compatibility_digest =>
                {
                    return Err(DeepSeekV4CheckpointStoreError::NamespaceCollision);
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
        Ok(DeepSeekV4LookupReport {
            candidates_examined: examined,
            corrupt_entries_removed: corrupt_removed,
            ..DeepSeekV4LookupReport::miss()
        })
    }

    /// Publish a prepared checkpoint captured earlier from a session that
    /// may no longer exist.
    pub fn publish_prepared(
        &self,
        prepared: &DeepSeekV4PreparedCheckpoint,
        max_record_bytes: u64,
    ) -> Result<DeepSeekV4PublishReport, DeepSeekV4CheckpointStoreError> {
        let context = DeepSeekV4StoreContext {
            compatibility_digest: prepared.snapshot.compatibility_digest(),
            codec_constraints: DeepSeekV4SnapshotCodecConstraints {
                config: &prepared.config,
                session_capacity: prepared.capacity,
                expected_model_content_id: prepared.model_content_id,
                max_record_bytes,
            },
            max_record_bytes,
        };
        self.publish(context, &prepared.snapshot)
    }

    fn publish_staged(
        &self,
        context: DeepSeekV4StoreContext<'_>,
        snapshot: &DeepSeekV4CausalSnapshot,
        digest: [u8; 32],
        final_path: &Path,
        staged: &StagedBlob,
        staging_cleanup: StagingCleanupReport,
    ) -> Result<DeepSeekV4PublishReport, DeepSeekV4CheckpointStoreError> {
        let mut repaired = false;
        for _ in 0..MAX_PUBLICATION_ATTEMPTS {
            if let Some(lease) = self.open_candidate(final_path)? {
                let valid =
                    match decode_causal_snapshot(&mut &lease.file, context.codec_constraints()) {
                        Ok(existing) => {
                            if !same_causal_state(&existing, snapshot)
                                || existing.next_position() != snapshot.next_position()
                                || snapshot_prefix_key(
                                    context.compatibility_digest.as_bytes(),
                                    &existing,
                                ) != digest
                            {
                                return Err(DeepSeekV4CheckpointStoreError::NamespaceCollision);
                            }
                            true
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
                DeepSeekV4CheckpointStoreError::StagedMetadata("staging path disappeared"),
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
                    return Err(DeepSeekV4CheckpointStoreError::PostMutationIo {
                        operation: "publish blob after eviction",
                        source,
                    });
                }
                return Err(source.into());
            }
            let staged_meta = metadata_nofollow(&staged.path)
                .map_err(|source| DeepSeekV4CheckpointStoreError::PostMutationIo {
                    operation: "stat staged blob after publication",
                    source,
                })?
                .ok_or(DeepSeekV4CheckpointStoreError::PostCommit(
                    "staged blob disappeared",
                ))?;
            let final_meta = metadata_nofollow(final_path)
                .map_err(|source| DeepSeekV4CheckpointStoreError::PostMutationIo {
                    operation: "stat final blob after publication",
                    source,
                })?
                .ok_or(DeepSeekV4CheckpointStoreError::PostCommit(
                    "final disappeared",
                ))?;
            let fd_stamp = FileStamp::from_metadata(&staged.file.metadata().map_err(|source| {
                DeepSeekV4CheckpointStoreError::PostMutationIo {
                    operation: "stat staged descriptor after publication",
                    source,
                }
            })?);
            let staged_stamp = FileStamp::from_metadata(&staged_meta);
            let final_stamp = FileStamp::from_metadata(&final_meta);
            validate_post_link_stamp(&fd_stamp, staged.encoded.record_bytes, &staged.opening)
                .map_err(|_| {
                    DeepSeekV4CheckpointStoreError::PostCommit("staged descriptor drifted")
                })?;
            validate_post_link_stamp(&staged_stamp, staged.encoded.record_bytes, &fd_stamp)
                .map_err(|_| DeepSeekV4CheckpointStoreError::PostCommit("staged path drifted"))?;
            validate_post_link_stamp(&final_stamp, staged.encoded.record_bytes, &fd_stamp)
                .map_err(|_| DeepSeekV4CheckpointStoreError::PostCommit("final blob drifted"))?;
            sync_directory(final_path.parent().expect("blob parent")).map_err(|source| {
                DeepSeekV4CheckpointStoreError::PostMutationIo {
                    operation: "sync published blob directory",
                    source,
                }
            })?;
            drop(lock);
            return Ok(DeepSeekV4PublishReport {
                outcome: if repaired {
                    PublishOutcome::RepairedCorrupt
                } else {
                    PublishOutcome::Published
                },
                blob_bytes: staged.encoded.record_bytes,
                managed_bytes_after: remaining_bytes
                    .checked_add(staged.encoded.record_bytes)
                    .ok_or(DeepSeekV4CheckpointStoreError::ManagedBytesOverflow)?,
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
        Err(DeepSeekV4CheckpointStoreError::ConcurrentChurn)
    }

    fn encode_staged(
        &self,
        temp_path: &Path,
        context: DeepSeekV4StoreContext<'_>,
        snapshot: &DeepSeekV4CausalSnapshot,
    ) -> Result<StagedBlob, DeepSeekV4CheckpointStoreError> {
        let mut file = self.namespace.create_staging_file(
            temp_path,
            self.staged_integrity == StagedIntegrityMode::Decode,
        )?;
        let opening = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&opening, 0, 1, None)?;
        let encoded = {
            let mut writer = BufWriter::new(&mut file);
            let encoded =
                encode_causal_snapshot(&mut writer, snapshot, context.codec_constraints())?;
            writer.flush()?;
            encoded
        };
        file.sync_all()?;
        let integrity_t0 = Instant::now();
        let synced = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&synced, encoded.record_bytes, 1, Some(&opening))?;
        if self.staged_integrity == StagedIntegrityMode::Decode {
            file.seek(SeekFrom::Start(0))?;
            let decoded = decode_causal_snapshot(&mut file, context.codec_constraints())?;
            if !same_causal_state(&decoded, snapshot)
                || decoded.next_position() != snapshot.next_position()
            {
                return Err(DeepSeekV4CheckpointStoreError::StagedValidation);
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
        request_tokens: &[u32],
        context: DeepSeekV4StoreContext<'_>,
    ) -> Result<Vec<Candidate>, DeepSeekV4CheckpointStoreError> {
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
        let digests = request_prefix_keys(
            context.compatibility_digest.as_bytes(),
            request_tokens,
            &lengths,
        );
        let mut candidates = Vec::new();
        for (path, parsed) in found {
            if parsed.matched_len == request_tokens.len() {
                continue;
            }
            if digests.get(&parsed.matched_len) == Some(&parsed.digest) {
                candidates.push(Candidate {
                    path,
                    matched_len: parsed.matched_len,
                    digest: parsed.digest,
                });
            }
        }
        candidates.sort_by(|a, b| {
            b.matched_len
                .cmp(&a.matched_len)
                .then_with(|| a.path.cmp(&b.path))
        });
        Ok(candidates)
    }

    fn ensure_blob_dir(
        &self,
        blob_dir: &Path,
    ) -> Result<StagingCleanupReport, DeepSeekV4CheckpointStoreError> {
        Ok(self.namespace.ensure_blob_dir(blob_dir)?)
    }

    fn admit_existing(
        &self,
        path: &Path,
        lease: &BlobLease,
        staged_integrity: StagedIntegrityReport,
        staging_cleanup: StagingCleanupReport,
    ) -> Result<Option<DeepSeekV4PublishReport>, DeepSeekV4CheckpointStoreError> {
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
        Ok(Some(DeepSeekV4PublishReport {
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

    fn open_candidate(
        &self,
        path: &Path,
    ) -> Result<Option<BlobLease>, DeepSeekV4CheckpointStoreError> {
        Ok(self.namespace.open_candidate(path)?)
    }

    fn remove_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<bool, DeepSeekV4CheckpointStoreError> {
        Ok(self.namespace.remove_if_same_inode(path, lease)?)
    }

    fn touch_if_same_inode(
        &self,
        path: &Path,
        lease: &BlobLease,
    ) -> Result<(), DeepSeekV4CheckpointStoreError> {
        Ok(self.namespace.touch_if_same_inode(path, lease)?)
    }

    #[cfg(test)]
    fn namespace_root(&self) -> PathBuf {
        self.namespace.namespace_root()
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

impl From<CheckpointFsError> for DeepSeekV4CheckpointStoreError {
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

/// A captured causal boundary bundled with everything publication needs, so
/// encoding and durability can run after the session (and its Metal state)
/// are gone. The DeepSeek V4 analog of the Qwen runtime's prepared
/// checkpoint: capture is cheap and happens at the boundary; the store I/O
/// happens whenever the caller chooses.
pub struct DeepSeekV4PreparedCheckpoint {
    snapshot: DeepSeekV4CausalSnapshot,
    config: DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    model_content_id: DeepSeekV4ModelContentId,
}

impl DeepSeekV4PreparedCheckpoint {
    pub fn next_position(&self) -> u32 {
        self.snapshot.next_position()
    }

    pub fn payload_bytes(&self) -> u64 {
        self.snapshot.payload_bytes()
    }
}

/// One durable probe-and-restore attempt against a session.
#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4DurableRestoreAttempt {
    /// `Some(len)` when a checkpoint was restored into the session.
    pub restored_prefix_len: Option<usize>,
    pub matched_prefix_len: usize,
    pub candidates_examined: usize,
    pub corrupt_entries_removed: usize,
    pub touched: bool,
    pub payload_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4DurableError {
    #[error("DeepSeek V4 session has no bound model-content identity for durable reuse")]
    UnboundIdentity,
    #[error(transparent)]
    Store(#[from] DeepSeekV4CheckpointStoreError),
    #[error("restore durable DeepSeek V4 causal snapshot: {0}")]
    Restore(DeepSeekV4MetalError),
}

impl DeepSeekV4Session {
    /// Probe `store` for the longest strict-prefix checkpoint of
    /// `request_tokens` and restore it into this ready session.
    ///
    /// A miss returns `Ok` with `restored_prefix_len: None`. Store and codec
    /// failures surface as `Store` errors; a hit whose restore fails
    /// pre-mutation surfaces as `Restore`, leaving the session ready for
    /// cold prefill. Callers own the fail-open policy.
    pub fn restore_durable_prefix(
        &mut self,
        store: &DeepSeekV4CheckpointStore,
        request_tokens: &[u32],
        max_record_bytes: u64,
    ) -> Result<DeepSeekV4DurableRestoreAttempt, DeepSeekV4DurableError> {
        let model_content_id = self
            .bound_model_content_id()
            .ok_or(DeepSeekV4DurableError::UnboundIdentity)?;
        let config = self.residency().config();
        let context = DeepSeekV4StoreContext {
            compatibility_digest: DeepSeekV4CompatibilityDigest::for_model(
                model_content_id,
                config,
            ),
            codec_constraints: DeepSeekV4SnapshotCodecConstraints {
                config,
                session_capacity: self.capacity(),
                expected_model_content_id: model_content_id,
                max_record_bytes,
            },
            max_record_bytes,
        };
        let mut lookup = store.lookup(context, request_tokens)?;
        let Some(snapshot) = lookup.snapshot.take() else {
            return Ok(DeepSeekV4DurableRestoreAttempt {
                restored_prefix_len: None,
                matched_prefix_len: lookup.matched_prefix_len,
                candidates_examined: lookup.candidates_examined,
                corrupt_entries_removed: lookup.corrupt_entries_removed,
                touched: lookup.touched,
                payload_bytes: 0,
            });
        };
        let payload_bytes = snapshot.payload_bytes();
        self.restore_causal_snapshot(&snapshot)
            .map_err(DeepSeekV4DurableError::Restore)?;
        Ok(DeepSeekV4DurableRestoreAttempt {
            restored_prefix_len: Some(lookup.restored_prefix_len),
            matched_prefix_len: lookup.matched_prefix_len,
            candidates_examined: lookup.candidates_examined,
            corrupt_entries_removed: lookup.corrupt_entries_removed,
            touched: lookup.touched,
            payload_bytes,
        })
    }

    /// Capture the session's current causal boundary for later publication.
    pub fn prepare_durable_checkpoint(
        &self,
    ) -> Result<DeepSeekV4PreparedCheckpoint, DeepSeekV4MetalError> {
        let model_content_id = self.bound_model_content_id().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 session has no bound model-content identity".into(),
            )
        })?;
        let snapshot = self.capture_causal_snapshot()?;
        Ok(DeepSeekV4PreparedCheckpoint {
            config: self.residency().config().clone(),
            capacity: self.capacity(),
            model_content_id,
            snapshot,
        })
    }
}

#[derive(Clone, Copy)]
pub struct DeepSeekV4StoreContext<'a> {
    pub compatibility_digest: DeepSeekV4CompatibilityDigest,
    pub codec_constraints: DeepSeekV4SnapshotCodecConstraints<'a>,
    pub max_record_bytes: u64,
}

impl DeepSeekV4StoreContext<'_> {
    fn validate(&self) -> Result<(), DeepSeekV4CheckpointStoreError> {
        if self.compatibility_digest
            != DeepSeekV4CompatibilityDigest::for_model(
                self.codec_constraints.expected_model_content_id,
                self.codec_constraints.config,
            )
        {
            return Err(DeepSeekV4CheckpointStoreError::CompatibilityMismatch);
        }
        Ok(())
    }

    fn codec_constraints(&self) -> DeepSeekV4SnapshotCodecConstraints<'_> {
        DeepSeekV4SnapshotCodecConstraints {
            max_record_bytes: self.max_record_bytes,
            ..self.codec_constraints
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4PublishReport {
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
pub struct DeepSeekV4LookupReport {
    pub snapshot: Option<DeepSeekV4CausalSnapshot>,
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub candidates_examined: usize,
    pub corrupt_entries_removed: usize,
    pub touched: bool,
}

impl DeepSeekV4LookupReport {
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
pub enum DeepSeekV4CheckpointStoreError {
    #[error("checkpoint store I/O: {0}")]
    Io(#[from] io::Error),
    #[error("checkpoint store codec: {0}")]
    Codec(#[from] DeepSeekV4SnapshotCodecError),
    #[error("DeepSeek V4 snapshot compatibility does not match store context")]
    CompatibilityMismatch,
    #[error("DeepSeek V4 checkpoint prefix key resolved to different causal state")]
    NamespaceCollision,
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

#[derive(Clone)]
struct Candidate {
    path: PathBuf,
    matched_len: usize,
    digest: [u8; 32],
}

struct StagedBlob {
    file: std::fs::File,
    path: PathBuf,
    encoded: DeepSeekV4EncodedSnapshot,
    opening: FileStamp,
    synced: FileStamp,
    integrity: StagedIntegrityReport,
}

struct ParsedBlobName {
    matched_len: usize,
    digest: [u8; 32],
}

fn snapshot_prefix_key(
    compatibility_id: &[u8; 32],
    snapshot: &DeepSeekV4CausalSnapshot,
) -> [u8; 32] {
    let mut hasher = prefix_key_hasher(compatibility_id);
    for &token in snapshot.prefix_tokens() {
        hasher.update(&token.to_le_bytes());
    }
    hasher.update(&(snapshot.next_position() as u64).to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn request_prefix_keys(
    compatibility_id: &[u8; 32],
    request_tokens: &[u32],
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
    snapshot: &DeepSeekV4CausalSnapshot,
    request_tokens: &[u32],
    matched_len: usize,
    compatibility_id: &[u8; 32],
    digest: &[u8; 32],
) -> bool {
    snapshot.next_position() as usize == matched_len
        && same_request_prefix(snapshot, request_tokens)
        && snapshot_prefix_key(compatibility_id, snapshot) == *digest
}

fn same_request_prefix(snapshot: &DeepSeekV4CausalSnapshot, request_tokens: &[u32]) -> bool {
    snapshot.prefix_tokens().iter().copied().eq(request_tokens
        .iter()
        .copied()
        .take(snapshot.next_position() as usize))
}

fn same_snapshot_prefix(a: &DeepSeekV4CausalSnapshot, b: &DeepSeekV4CausalSnapshot) -> bool {
    a.prefix_tokens() == b.prefix_tokens()
}

fn same_causal_state(a: &DeepSeekV4CausalSnapshot, b: &DeepSeekV4CausalSnapshot) -> bool {
    same_snapshot_prefix(a, b)
        && a.model_content_id() == b.model_content_id()
        && a.compatibility_digest() == b.compatibility_digest()
        && a.next_position() == b.next_position()
        && a.causal_digest() == b.causal_digest()
}

fn blob_name(matched_len: usize, digest: &[u8; 32]) -> String {
    format!("{matched_len}-{}.{}", hex(digest), BLOB_EXTENSION)
}

fn parse_blob_name(name: &std::ffi::OsStr) -> Option<ParsedBlobName> {
    let name = name.to_str()?;
    let stem = name.strip_suffix(&format!(".{BLOB_EXTENSION}"))?;
    let mut parts = stem.split('-');
    let length_text = parts.next()?;
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
        digest,
    })
}

fn codec_error_proves_invalid_blob(error: &DeepSeekV4SnapshotCodecError) -> bool {
    match error {
        DeepSeekV4SnapshotCodecError::InvalidHeader(_)
        | DeepSeekV4SnapshotCodecError::CompatibilityMismatch
        | DeepSeekV4SnapshotCodecError::ModelIdentityMismatch
        | DeepSeekV4SnapshotCodecError::ArithmeticOverflow { .. }
        | DeepSeekV4SnapshotCodecError::SectionLength { .. }
        | DeepSeekV4SnapshotCodecError::DigestMismatch
        | DeepSeekV4SnapshotCodecError::TrailingBytes => true,
        DeepSeekV4SnapshotCodecError::Snapshot(_) => true,
        DeepSeekV4SnapshotCodecError::Io(error) => error.kind() == io::ErrorKind::UnexpectedEof,
        DeepSeekV4SnapshotCodecError::UnsupportedByteOrder
        | DeepSeekV4SnapshotCodecError::RecordBudgetExceeded { .. }
        | DeepSeekV4SnapshotCodecError::AllocationFailed { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_fs::unique_temp_path;
    use crate::metal::MetalContext;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qwen-dsv4-checkpoint-store-{label}-{}-{sequence}",
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

    #[test]
    fn namespace_is_versioned_and_collision_safe_with_qwen() {
        let store = DeepSeekV4CheckpointStore::new("/tmp/checkpoints", 1024);
        assert_eq!(
            store.namespace_root(),
            Path::new("/tmp/checkpoints/dsv4-v1")
        );
        assert_ne!(store.namespace_root(), Path::new("/tmp/checkpoints/v1"));
        assert_eq!(
            store.identity_cache().root(),
            Path::new("/tmp/checkpoints/dsv4-v1/identity")
        );
    }

    #[test]
    fn blob_names_are_canonical_and_bind_length_and_digest() {
        let digest = [0xab; 32];
        let name = blob_name(12, &digest);
        let parsed = parse_blob_name(std::ffi::OsStr::new(&name)).unwrap();
        assert_eq!(parsed.matched_len, 12);
        assert_eq!(parsed.digest, digest);
        for invalid in [
            "012-abababababababababababababababababababababababababababababababab.dsv4cp",
            "12-ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB.dsv4cp",
            "12-ab.dsv4cp",
            "12-abababababababababababababababababababababababababababababababab.qcp",
        ] {
            assert!(parse_blob_name(std::ffi::OsStr::new(invalid)).is_none());
        }
    }

    #[test]
    fn shared_staging_cleanup_is_wired_into_deepseek_publication() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("staging-cleanup");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let fixture = Fixture::new(&ctx, 4, 17);
        let blob_dir = store.blob_dir(fixture.snapshot.compatibility_digest().as_bytes());
        store.ensure_blob_dir(&blob_dir).unwrap();
        let abandoned = unique_temp_path(&blob_dir, &[0x42; 32]);
        let mut file = store
            .namespace
            .create_staging_file(&abandoned, true)
            .unwrap();
        file.write_all(b"abandoned-deepseek-staging").unwrap();
        drop(file);

        let report = store
            .publish(fixture.context(1 << 30), &fixture.snapshot)
            .unwrap();
        assert_eq!(report.staging_entries_removed, 1);
        assert!(report.staging_allocated_bytes_reclaimed >= 26);
        assert!(!abandoned.exists());
    }

    #[test]
    fn non_corruption_codec_failures_are_preserved() {
        for error in [
            DeepSeekV4SnapshotCodecError::RecordBudgetExceeded {
                record_bytes: 2,
                max_record_bytes: 1,
            },
            DeepSeekV4SnapshotCodecError::AllocationFailed {
                section: "test",
                bytes: 1,
            },
            DeepSeekV4SnapshotCodecError::Io(io::Error::other("transient")),
        ] {
            assert!(!codec_error_proves_invalid_blob(&error));
        }
        assert!(codec_error_proves_invalid_blob(
            &DeepSeekV4SnapshotCodecError::DigestMismatch
        ));
        assert!(codec_error_proves_invalid_blob(
            &DeepSeekV4SnapshotCodecError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated",
            ))
        ));
    }

    #[test]
    fn store_publishes_and_restores_longest_strict_prefix() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("basic");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let (config, capacity, model_content_id, short) =
            DeepSeekV4CausalSnapshot::synthetic_test_fixture(&ctx, 4, 17);
        let (_, _, _, long) = DeepSeekV4CausalSnapshot::synthetic_test_fixture(&ctx, 8, 23);
        let compatibility_digest = short.compatibility_digest();
        let context = DeepSeekV4StoreContext {
            compatibility_digest,
            codec_constraints: DeepSeekV4SnapshotCodecConstraints {
                config: &config,
                session_capacity: capacity,
                expected_model_content_id: model_content_id,
                max_record_bytes: 1 << 30,
            },
            max_record_bytes: 1 << 30,
        };

        assert!(!store.has_managed_blobs().unwrap());
        assert_eq!(
            store.publish(context, &short).unwrap().outcome,
            PublishOutcome::Published
        );
        assert_eq!(
            store.publish(context, &long).unwrap().outcome,
            PublishOutcome::Published
        );
        assert!(store.has_managed_blobs().unwrap());

        let mut extension = long.prefix_tokens().to_vec();
        extension.push(99);
        let hit = store.lookup(context, &extension).unwrap();
        assert_eq!(hit.matched_prefix_len, 8);
        assert_eq!(hit.restored_prefix_len, 8);
        assert!(!hit.exact);
        assert_eq!(hit.snapshot.unwrap().prefix_tokens(), long.prefix_tokens());

        let exact = store.lookup(context, long.prefix_tokens()).unwrap();
        assert_eq!(exact.matched_prefix_len, 4);
        assert!(!exact.exact);
        assert_eq!(
            exact.snapshot.unwrap().prefix_tokens(),
            short.prefix_tokens()
        );

        let mut short_extension = short.prefix_tokens().to_vec();
        short_extension.push(77);
        let hit = store.lookup(context, &short_extension).unwrap();
        assert_eq!(hit.matched_prefix_len, 4);
    }

    struct Fixture {
        config: crate::deepseek_v4::DeepSeekV4Config,
        capacity: crate::deepseek_v4_metal::DeepSeekV4SessionCapacity,
        model_content_id: crate::deepseek_v4_metal::DeepSeekV4ModelContentId,
        snapshot: DeepSeekV4CausalSnapshot,
    }

    impl Fixture {
        fn new(ctx: &MetalContext, next_position: u32, seed: u32) -> Self {
            let (config, capacity, model_content_id, snapshot) =
                DeepSeekV4CausalSnapshot::synthetic_test_fixture(ctx, next_position, seed);
            Self {
                config,
                capacity,
                model_content_id,
                snapshot,
            }
        }

        fn context(&self, max_record_bytes: u64) -> DeepSeekV4StoreContext<'_> {
            DeepSeekV4StoreContext {
                compatibility_digest: self.snapshot.compatibility_digest(),
                codec_constraints: DeepSeekV4SnapshotCodecConstraints {
                    config: &self.config,
                    session_capacity: self.capacity,
                    expected_model_content_id: self.model_content_id,
                    max_record_bytes,
                },
                max_record_bytes,
            }
        }

        fn blob_path(&self, store: &DeepSeekV4CheckpointStore) -> PathBuf {
            let digest = snapshot_prefix_key(
                self.snapshot.compatibility_digest().as_bytes(),
                &self.snapshot,
            );
            store
                .blob_dir(self.snapshot.compatibility_digest().as_bytes())
                .join(blob_name(self.snapshot.next_position() as usize, &digest))
        }
    }

    fn managed_blob_count(store: &DeepSeekV4CheckpointStore) -> usize {
        scan_managed_blobs(&store.blobs_root(), is_managed_blob_name)
            .unwrap()
            .blobs
            .len()
    }

    #[test]
    fn budget_eviction_is_lru_and_respects_touch_freshness() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("evict");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let short = Fixture::new(&ctx, 4, 17);
        let long = Fixture::new(&ctx, 8, 23);
        let short_bytes = store
            .publish(short.context(1 << 30), &short.snapshot)
            .unwrap()
            .blob_bytes;
        std::thread::sleep(std::time::Duration::from_millis(5));
        store
            .publish(long.context(1 << 30), &long.snapshot)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        // A lookup hit LRU-touches the short blob, making the long blob the
        // eviction victim despite its later publication.
        let mut extension = short.snapshot.prefix_tokens().to_vec();
        extension.push(77);
        assert!(
            store
                .lookup(short.context(1 << 30), &extension)
                .unwrap()
                .touched
        );

        let constrained = DeepSeekV4CheckpointStore::new(&temp.0, short_bytes);
        let report = constrained
            .publish(short.context(1 << 30), &short.snapshot)
            .unwrap();
        assert_eq!(report.outcome, PublishOutcome::ExistingValid);
        assert_eq!(report.evicted_entries, 1);
        assert_eq!(report.managed_bytes_after, short_bytes);
        assert!(short.blob_path(&store).exists());
        assert!(!long.blob_path(&store).exists());
        assert_eq!(managed_blob_count(&store), 1);
    }

    #[test]
    fn oversized_publication_is_rejected_without_namespace_mutation() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("oversized");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1024);
        let fixture = Fixture::new(&ctx, 4, 17);
        assert!(matches!(
            store.publish(fixture.context(1 << 30), &fixture.snapshot),
            Err(DeepSeekV4CheckpointStoreError::OversizedBlob { .. })
        ));
        assert_eq!(managed_blob_count(&store), 0);
        // Staging temp files must not survive the failed publication.
        let compat_dir = store.blob_dir(fixture.snapshot.compatibility_digest().as_bytes());
        let leftovers = std::fs::read_dir(&compat_dir)
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn caller_record_budget_failure_preserves_the_valid_blob() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("caller-budget");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let fixture = Fixture::new(&ctx, 4, 17);
        let published = store
            .publish(fixture.context(1 << 30), &fixture.snapshot)
            .unwrap();
        let mut extension = fixture.snapshot.prefix_tokens().to_vec();
        extension.push(99);

        assert!(matches!(
            store.lookup(fixture.context(published.blob_bytes - 1), &extension),
            Err(DeepSeekV4CheckpointStoreError::Codec(
                DeepSeekV4SnapshotCodecError::RecordBudgetExceeded { .. }
            ))
        ));
        assert!(fixture.blob_path(&store).exists());
        assert!(
            store
                .lookup(fixture.context(1 << 30), &extension)
                .unwrap()
                .snapshot
                .is_some()
        );
    }

    #[test]
    fn corrupt_blob_is_self_healed_and_lookup_falls_back() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("heal");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let short = Fixture::new(&ctx, 4, 17);
        let long = Fixture::new(&ctx, 8, 23);
        store
            .publish(short.context(1 << 30), &short.snapshot)
            .unwrap();
        store
            .publish(long.context(1 << 30), &long.snapshot)
            .unwrap();

        // Flip one payload byte in the longer record: the whole-record digest
        // no longer verifies, so lookup deletes it and falls back.
        let long_path = long.blob_path(&store);
        let mut bytes = std::fs::read(&long_path).unwrap();
        let offset = bytes.len() - 64;
        bytes[offset] ^= 0x5a;
        std::fs::write(&long_path, &bytes).unwrap();

        let mut extension = long.snapshot.prefix_tokens().to_vec();
        extension.push(99);
        let report = store.lookup(long.context(1 << 30), &extension).unwrap();
        assert_eq!(report.corrupt_entries_removed, 1);
        assert_eq!(report.matched_prefix_len, 4);
        assert!(!long_path.exists());
        assert!(short.blob_path(&store).exists());
    }

    #[test]
    fn foreign_entry_at_managed_key_is_rejected() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("foreign");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let fixture = Fixture::new(&ctx, 4, 17);
        store
            .publish(fixture.context(1 << 30), &fixture.snapshot)
            .unwrap();
        let path = fixture.blob_path(&store);
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(temp.0.join("elsewhere"), &path).unwrap();

        let mut extension = fixture.snapshot.prefix_tokens().to_vec();
        extension.push(99);
        assert!(matches!(
            store.lookup(fixture.context(1 << 30), &extension),
            Err(DeepSeekV4CheckpointStoreError::ForeignEntryAtKey(_))
        ));
    }

    #[test]
    fn republishing_different_causal_state_for_same_prefix_is_a_collision() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("collision");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let original = Fixture::new(&ctx, 4, 17);
        let variant = Fixture::new(&ctx, 4, 99);
        // The synthetic fixture derives tokens from positions, so both
        // snapshots name the same blob while carrying different causal bits —
        // the determinism violation the store refuses to repair silently.
        assert_eq!(
            original.snapshot.prefix_tokens(),
            variant.snapshot.prefix_tokens()
        );
        assert_ne!(
            original.snapshot.causal_digest(),
            variant.snapshot.causal_digest()
        );

        store
            .publish(original.context(1 << 30), &original.snapshot)
            .unwrap();
        assert!(matches!(
            store.publish(variant.context(1 << 30), &variant.snapshot),
            Err(DeepSeekV4CheckpointStoreError::NamespaceCollision)
        ));
        let mut extension = original.snapshot.prefix_tokens().to_vec();
        extension.push(99);
        let survivor = store
            .lookup(original.context(1 << 30), &extension)
            .unwrap()
            .snapshot
            .unwrap();
        assert_eq!(survivor.causal_digest(), original.snapshot.causal_digest());
    }

    #[test]
    fn prepared_checkpoints_publish_after_their_session_is_gone() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("prepared");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let fixture = Fixture::new(&ctx, 4, 17);
        let prepared = DeepSeekV4PreparedCheckpoint {
            snapshot: fixture.snapshot.clone(),
            config: fixture.config.clone(),
            capacity: fixture.capacity,
            model_content_id: fixture.model_content_id,
        };
        assert_eq!(prepared.next_position(), 4);

        let published = store.publish_prepared(&prepared, 1 << 30).unwrap();
        assert_eq!(published.outcome, PublishOutcome::Published);
        assert_eq!(
            store.publish_prepared(&prepared, 1 << 30).unwrap().outcome,
            PublishOutcome::ExistingValid
        );

        // The prepared record is byte-identical to a direct publication.
        let mut extension = fixture.snapshot.prefix_tokens().to_vec();
        extension.push(99);
        let survivor = store
            .lookup(fixture.context(1 << 30), &extension)
            .unwrap()
            .snapshot
            .unwrap();
        assert_eq!(survivor.causal_digest(), fixture.snapshot.causal_digest());
    }

    #[test]
    fn concurrent_publishers_converge_on_one_immutable_blob() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let temp = TestDir::new("concurrent");
        let store = DeepSeekV4CheckpointStore::new(&temp.0, 1 << 30);
        let fixture = Fixture::new(&ctx, 4, 17);
        let outcomes = std::thread::scope(|scope| {
            let handles = [
                scope.spawn(|| {
                    store
                        .clone()
                        .publish(fixture.context(1 << 30), &fixture.snapshot)
                }),
                scope.spawn(|| {
                    store
                        .clone()
                        .publish(fixture.context(1 << 30), &fixture.snapshot)
                }),
            ];
            handles.map(|handle| handle.join().unwrap().unwrap())
        });
        for report in &outcomes {
            assert!(matches!(
                report.outcome,
                PublishOutcome::Published | PublishOutcome::ExistingValid
            ));
        }
        assert!(
            outcomes
                .iter()
                .any(|report| report.outcome == PublishOutcome::Published)
        );
        assert_eq!(managed_blob_count(&store), 1);
        assert!(fixture.blob_path(&store).exists());
    }
}
