//! Bounded, catalog-free durable checkpoint blob store, generic over the
//! checkpoint payload (Qwen `SessionSnapshot`, DeepSeek V4 causal snapshot).
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
//!
//! What differs per family lives in [`DurablePayload`]: the codec, blob
//! naming and prefix keys, which records may serve a request, and whether a
//! mismatched record at a final key is repaired or reported as a collision.

use crate::checkpoint_fs::{
    BlobLease, CheckpointFsError, FileStamp, StagingCleanupReport, StoreNamespace, evict_to_fit,
    has_managed_blob, metadata_nofollow, path_exists_nofollow, require_real_directory_if_exists,
    scan_managed_blobs, sync_directory, unique_temp_path, validate_post_link_stamp,
    validate_staged_stamp,
};
use crate::checkpoint_identity::CheckpointIdentityCache;
use std::collections::{BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs::FileTimes;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

const MAX_PUBLICATION_ATTEMPTS: usize = 8;

/// How a decoded lookup candidate relates to the request it was found for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LookupVerdict {
    /// Serves the request.
    Match,
    /// A valid record for this key that describes different state; an error.
    Collision,
    /// Does not describe the key it is stored under; removed as corrupt.
    Mismatch,
}

/// A family's checkpoint payload as the store sees it.
pub trait DurablePayload: Sized {
    /// Request token type.
    type Token: Copy;
    /// Everything encode/decode needs to validate a record for this process.
    type Context<'a>: Copy;
    type CodecError: std::error::Error + Send + Sync + 'static;
    /// Distinguishes records of one prefix (Qwen: pending token / logits).
    type Mode: Copy + Eq + std::fmt::Debug;
    /// Family name carried by store errors ("DeepSeek V4 snapshot ...").
    const FAMILY_LABEL: &'static str;
    /// Namespace directory version; records of other versions are invisible.
    const NAMESPACE_VERSION: &'static str;
    /// A final key already holding a valid record with different state is a
    /// hard error (true) rather than repaired by replacement (false).
    const MISMATCHED_EXISTING_IS_COLLISION: bool;

    fn compatibility_id(context: &Self::Context<'_>) -> [u8; 32];
    /// Refuse a publication this context cannot describe.
    fn check_publish(
        &self,
        context: &Self::Context<'_>,
    ) -> Result<(), DurableStoreError<Self::CodecError>>;
    /// Refuse a lookup under an invalid context.
    fn check_lookup(context: &Self::Context<'_>)
    -> Result<(), DurableStoreError<Self::CodecError>>;
    fn matched_len(&self) -> usize;
    fn mode(&self) -> Self::Mode;
    fn prefix_key(&self, compatibility_id: &[u8; 32]) -> [u8; 32];
    fn request_prefix_keys(
        compatibility_id: &[u8; 32],
        request_tokens: &[Self::Token],
        lengths: &BTreeSet<usize>,
    ) -> HashMap<usize, [u8; 32]>;
    fn blob_name(matched_len: usize, mode: Self::Mode, digest: &[u8; 32]) -> String;
    /// `(matched_len, mode, digest)` of a managed blob name.
    fn parse_blob_name(name: &OsStr) -> Option<(usize, Self::Mode, [u8; 32])>;
    /// Whether a record of `mode` may serve a request, given whether it
    /// covers the whole request, and its rank among records of equal length
    /// (lower is tried first).
    fn candidate_rank(mode: Self::Mode, covers_request: bool) -> Option<usize>;
    /// Committed position a hit restores to.
    fn restored_len(mode: Self::Mode, matched_len: usize) -> usize;
    fn lookup_verdict(
        &self,
        request_tokens: &[Self::Token],
        matched_len: usize,
        mode: Self::Mode,
        context: &Self::Context<'_>,
        digest: &[u8; 32],
    ) -> LookupVerdict;
    /// The same checkpoint state (staged verification, existing final key).
    fn same_checkpoint(&self, other: &Self) -> bool;
    /// Exact size of the record `encode` would write, with its validation.
    fn encoded_record_bytes(&self, context: Self::Context<'_>) -> Result<u64, Self::CodecError>;
    /// Write the record; returns its size in bytes.
    fn encode<W: Write>(
        &self,
        writer: &mut W,
        context: Self::Context<'_>,
    ) -> Result<u64, Self::CodecError>;
    fn decode<R: Read>(
        reader: &mut R,
        context: Self::Context<'_>,
    ) -> Result<Self, Self::CodecError>;
    /// The record is invalid here and everywhere; remove it.
    fn codec_error_proves_invalid(error: &Self::CodecError) -> bool;
    /// An I/O failure, not a property of the record.
    fn codec_error_is_io(error: &Self::CodecError) -> bool;
}

#[derive(Debug)]
pub struct DurableStore<P> {
    namespace: StoreNamespace,
    max_managed_blob_bytes: u64,
    staged_integrity: StagedIntegrityMode,
    staged_integrity_explicit: bool,
    payload: PhantomData<fn() -> P>,
}

impl<P> Clone for DurableStore<P> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            max_managed_blob_bytes: self.max_managed_blob_bytes,
            staged_integrity: self.staged_integrity,
            staged_integrity_explicit: self.staged_integrity_explicit,
            payload: PhantomData,
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
pub struct LookupReport<S> {
    pub snapshot: Option<S>,
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub candidates_examined: usize,
    pub corrupt_entries_removed: usize,
    /// Valid records skipped because this process cannot use them.
    pub unusable_skipped: usize,
    pub touched: bool,
}

impl<S> LookupReport<S> {
    fn miss() -> Self {
        Self {
            snapshot: None,
            matched_prefix_len: 0,
            restored_prefix_len: 0,
            exact: false,
            candidates_examined: 0,
            corrupt_entries_removed: 0,
            unusable_skipped: 0,
            touched: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DurableStoreError<C: std::error::Error + 'static> {
    #[error("checkpoint store I/O: {0}")]
    Io(#[from] io::Error),
    #[error("checkpoint store codec: {0}")]
    Codec(#[source] C),
    #[error("{family} snapshot compatibility does not match store context")]
    CompatibilityMismatch { family: &'static str },
    #[error("{family} checkpoint prefix key resolved to different causal state")]
    NamespaceCollision { family: &'static str },
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

impl<C: std::error::Error + 'static> From<CheckpointFsError> for DurableStoreError<C> {
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

struct Candidate<M> {
    path: PathBuf,
    matched_len: usize,
    mode: M,
    digest: [u8; 32],
}

pub(crate) struct StagedBlob {
    pub(crate) file: std::fs::File,
    pub(crate) path: PathBuf,
    pub(crate) record_bytes: u64,
    opening: FileStamp,
    synced: FileStamp,
    pub(crate) integrity: StagedIntegrityReport,
}

type StoreResult<P, T> = Result<T, DurableStoreError<<P as DurablePayload>::CodecError>>;

impl<P: DurablePayload> DurableStore<P> {
    pub fn new(root: impl Into<PathBuf>, max_managed_blob_bytes: u64) -> Self {
        Self {
            namespace: StoreNamespace::new(root, P::NAMESPACE_VERSION),
            max_managed_blob_bytes,
            staged_integrity: StagedIntegrityMode::Decode,
            staged_integrity_explicit: false,
            payload: PhantomData,
        }
    }

    pub fn with_staged_integrity(
        root: impl Into<PathBuf>,
        max_managed_blob_bytes: u64,
        staged_integrity: StagedIntegrityMode,
    ) -> Self {
        Self {
            namespace: StoreNamespace::new(root, P::NAMESPACE_VERSION),
            max_managed_blob_bytes,
            staged_integrity,
            staged_integrity_explicit: true,
            payload: PhantomData,
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
    pub fn has_managed_blobs(&self) -> StoreResult<P, bool> {
        let _lock = self.namespace.lock_shared()?;
        Ok(has_managed_blob(
            &self.namespace.blobs_root(),
            is_managed_blob_name::<P>,
        )?)
    }

    pub fn publish(&self, context: P::Context<'_>, snapshot: &P) -> StoreResult<P, PublishReport> {
        snapshot.check_publish(&context)?;
        let compatibility_id = P::compatibility_id(&context);
        let mode = snapshot.mode();
        let digest = snapshot.prefix_key(&compatibility_id);
        let blob_dir = self.namespace.blob_dir(&compatibility_id);
        // One publisher at a time from the space check through publication.
        let _writer = self.namespace.lock_writer()?;
        let staging_cleanup = self.namespace.ensure_blob_dir(&blob_dir)?;
        // Size and validate the record exactly (the codec enforces the record
        // budget) and check the store budget before reclaiming any space.
        let record_bytes = snapshot
            .encoded_record_bytes(context)
            .map_err(DurableStoreError::Codec)?;
        if record_bytes > self.max_managed_blob_bytes {
            return Err(DurableStoreError::OversizedBlob {
                blob_bytes: record_bytes,
                max_managed_blob_bytes: self.max_managed_blob_bytes,
            });
        }
        self.namespace
            .ensure_volume_space(record_bytes, is_managed_blob_name::<P>)?;
        let final_path = blob_dir.join(P::blob_name(snapshot.matched_len(), mode, &digest));
        let temp_path = unique_temp_path(&blob_dir, &digest);
        let staged = match self.encode_staged(&temp_path, context, snapshot) {
            Ok(staged) => staged,
            Err(error) => {
                let _ = std::fs::remove_file(&temp_path);
                return Err(error);
            }
        };
        if staged.record_bytes > self.max_managed_blob_bytes {
            let _ = std::fs::remove_file(&temp_path);
            return Err(DurableStoreError::OversizedBlob {
                blob_bytes: staged.record_bytes,
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
        context: P::Context<'_>,
        request_tokens: &[P::Token],
    ) -> StoreResult<P, LookupReport<P>> {
        self.lookup_filtered(context, request_tokens, |_, _| true)
    }

    /// [`Self::lookup`], consulting `admit(matched_len, blob_bytes)` before
    /// decoding each candidate (longest first). A rejected candidate is
    /// skipped without reading it, so callers can require a longer match
    /// than they already hold or refuse records they cannot afford to load.
    pub fn lookup_filtered(
        &self,
        context: P::Context<'_>,
        request_tokens: &[P::Token],
        mut admit: impl FnMut(usize, u64) -> bool,
    ) -> StoreResult<P, LookupReport<P>> {
        P::check_lookup(&context)?;
        if request_tokens.is_empty() {
            return Ok(LookupReport::miss());
        }
        let compatibility_id = P::compatibility_id(&context);
        let blob_dir = self.namespace.blob_dir(&compatibility_id);
        let candidates = self.discover_candidates(&blob_dir, request_tokens, &compatibility_id)?;
        let mut examined = 0usize;
        let mut corrupt_removed = 0usize;
        let mut unusable_skipped = 0usize;
        for candidate in candidates {
            examined += 1;
            let Some(lease) = self.namespace.open_candidate_for_lookup(&candidate.path)? else {
                continue;
            };
            if !admit(candidate.matched_len, lease.size) {
                continue;
            }
            match P::decode(&mut &lease.file, context) {
                Ok(snapshot) => match snapshot.lookup_verdict(
                    request_tokens,
                    candidate.matched_len,
                    candidate.mode,
                    &context,
                    &candidate.digest,
                ) {
                    LookupVerdict::Match => {
                        let touched = self
                            .namespace
                            .touch_if_same_inode(&candidate.path, &lease)
                            .is_ok();
                        return Ok(LookupReport {
                            snapshot: Some(snapshot),
                            matched_prefix_len: candidate.matched_len,
                            restored_prefix_len: P::restored_len(
                                candidate.mode,
                                candidate.matched_len,
                            ),
                            exact: candidate.matched_len == request_tokens.len(),
                            candidates_examined: examined,
                            corrupt_entries_removed: corrupt_removed,
                            unusable_skipped,
                            touched,
                        });
                    }
                    LookupVerdict::Collision => {
                        return Err(DurableStoreError::NamespaceCollision {
                            family: P::FAMILY_LABEL,
                        });
                    }
                    LookupVerdict::Mismatch => {
                        if self
                            .namespace
                            .remove_if_same_inode_for_lookup(&candidate.path, &lease)?
                        {
                            corrupt_removed += 1;
                        }
                    }
                },
                Err(error) if P::codec_error_proves_invalid(&error) => {
                    if self
                        .namespace
                        .remove_if_same_inode_for_lookup(&candidate.path, &lease)?
                    {
                        corrupt_removed += 1;
                    }
                }
                // Valid but unusable here (over the current record budget,
                // context capacity, or allocation): keep it for a process
                // that can load it and try the next, shorter candidate.
                Err(error) if !P::codec_error_is_io(&error) => {
                    unusable_skipped += 1;
                }
                Err(error) => return Err(DurableStoreError::Codec(error)),
            }
        }
        Ok(LookupReport {
            candidates_examined: examined,
            corrupt_entries_removed: corrupt_removed,
            unusable_skipped,
            ..LookupReport::miss()
        })
    }

    fn publish_staged(
        &self,
        context: P::Context<'_>,
        snapshot: &P,
        digest: [u8; 32],
        final_path: &Path,
        staged: &StagedBlob,
        staging_cleanup: StagingCleanupReport,
    ) -> StoreResult<P, PublishReport> {
        let compatibility_id = P::compatibility_id(&context);
        let mut repaired = false;
        for _ in 0..MAX_PUBLICATION_ATTEMPTS {
            if let Some(lease) = self.namespace.open_candidate(final_path)? {
                let valid = match P::decode(&mut &lease.file, context) {
                    Ok(existing) => {
                        let same = existing.same_checkpoint(snapshot)
                            && existing.prefix_key(&compatibility_id) == digest;
                        if !same && P::MISMATCHED_EXISTING_IS_COLLISION {
                            return Err(DurableStoreError::NamespaceCollision {
                                family: P::FAMILY_LABEL,
                            });
                        }
                        same
                    }
                    Err(error) if P::codec_error_proves_invalid(&error) => false,
                    Err(error) => return Err(DurableStoreError::Codec(error)),
                };
                if valid {
                    if let Some(report) =
                        self.admit_existing(final_path, &lease, staged.integrity, staging_cleanup)?
                    {
                        return Ok(report);
                    }
                    continue;
                }
                repaired |= self.namespace.remove_if_same_inode(final_path, &lease)?;
                continue;
            }

            let lock = self.namespace.lock_exclusive()?;
            if path_exists_nofollow(final_path)? {
                drop(lock);
                continue;
            }
            let before =
                scan_managed_blobs(&self.namespace.blobs_root(), is_managed_blob_name::<P>)?;
            let (evicted_entries, evicted_bytes, remaining_bytes) = evict_to_fit(
                before,
                staged.record_bytes,
                self.max_managed_blob_bytes,
                Some(final_path),
            )?;
            let source_meta = metadata_nofollow(&staged.path)?.ok_or(
                DurableStoreError::StagedMetadata("staging path disappeared"),
            )?;
            let source_stamp = FileStamp::from_metadata(&source_meta);
            validate_staged_stamp(&source_stamp, staged.record_bytes, 1, Some(&staged.synced))?;
            if let Err(source) = std::fs::hard_link(&staged.path, final_path) {
                if evicted_entries > 0 {
                    return Err(DurableStoreError::PostMutationIo {
                        operation: "publish blob after eviction",
                        source,
                    });
                }
                return Err(source.into());
            }
            let staged_meta = metadata_nofollow(&staged.path)
                .map_err(|source| DurableStoreError::PostMutationIo {
                    operation: "stat staged blob after publication",
                    source,
                })?
                .ok_or(DurableStoreError::PostCommit("staged blob disappeared"))?;
            let final_meta = metadata_nofollow(final_path)
                .map_err(|source| DurableStoreError::PostMutationIo {
                    operation: "stat final blob after publication",
                    source,
                })?
                .ok_or(DurableStoreError::PostCommit("final disappeared"))?;
            let fd_stamp = FileStamp::from_metadata(&staged.file.metadata().map_err(|source| {
                DurableStoreError::PostMutationIo {
                    operation: "stat staged descriptor after publication",
                    source,
                }
            })?);
            let staged_stamp = FileStamp::from_metadata(&staged_meta);
            let final_stamp = FileStamp::from_metadata(&final_meta);
            validate_post_link_stamp(&fd_stamp, staged.record_bytes, &staged.opening)
                .map_err(|_| DurableStoreError::PostCommit("staged descriptor drifted"))?;
            validate_post_link_stamp(&staged_stamp, staged.record_bytes, &fd_stamp)
                .map_err(|_| DurableStoreError::PostCommit("staged path drifted"))?;
            validate_post_link_stamp(&final_stamp, staged.record_bytes, &fd_stamp)
                .map_err(|_| DurableStoreError::PostCommit("final blob drifted"))?;
            sync_directory(final_path.parent().expect("blob parent")).map_err(|source| {
                DurableStoreError::PostMutationIo {
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
                    .ok_or(DurableStoreError::ManagedBytesOverflow)?,
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
        Err(DurableStoreError::ConcurrentChurn)
    }

    pub(crate) fn encode_staged(
        &self,
        temp_path: &Path,
        context: P::Context<'_>,
        snapshot: &P,
    ) -> StoreResult<P, StagedBlob> {
        let mut file = self.namespace.create_staging_file(
            temp_path,
            self.staged_integrity == StagedIntegrityMode::Decode,
        )?;
        let opening = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&opening, 0, 1, None)?;
        let record_bytes = {
            let mut writer = BufWriter::new(&mut file);
            let record_bytes = snapshot
                .encode(&mut writer, context)
                .map_err(DurableStoreError::Codec)?;
            writer.flush()?;
            record_bytes
        };
        file.sync_all()?;
        let integrity_t0 = Instant::now();
        let synced = FileStamp::from_metadata(&file.metadata()?);
        validate_staged_stamp(&synced, record_bytes, 1, Some(&opening))?;
        if self.staged_integrity == StagedIntegrityMode::Decode {
            file.seek(SeekFrom::Start(0))?;
            let decoded = P::decode(&mut file, context).map_err(DurableStoreError::Codec)?;
            if !decoded.same_checkpoint(snapshot) {
                return Err(DurableStoreError::StagedValidation);
            }
        }
        Ok(StagedBlob {
            file,
            path: temp_path.to_path_buf(),
            record_bytes,
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
        request_tokens: &[P::Token],
        compatibility_id: &[u8; 32],
    ) -> StoreResult<P, Vec<Candidate<P::Mode>>> {
        let lock = self.namespace.lock_shared_for_lookup()?;
        let mut found = Vec::new();
        let mut lengths = BTreeSet::new();
        if !require_real_directory_if_exists(&self.namespace.blobs_root())?
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
            let Some(parsed) = P::parse_blob_name(&entry.file_name()) else {
                continue;
            };
            if parsed.0 <= request_tokens.len() {
                lengths.insert(parsed.0);
                found.push((entry.path(), parsed));
            }
        }
        drop(lock);
        let digests = P::request_prefix_keys(compatibility_id, request_tokens, &lengths);
        let mut ranked = Vec::new();
        for (path, (matched_len, mode, digest)) in found {
            let Some(rank) = P::candidate_rank(mode, matched_len == request_tokens.len()) else {
                continue;
            };
            if digests.get(&matched_len) == Some(&digest) {
                ranked.push((
                    rank,
                    Candidate {
                        path,
                        matched_len,
                        mode,
                        digest,
                    },
                ));
            }
        }
        ranked.sort_by(|(a_rank, a), (b_rank, b)| {
            b.matched_len
                .cmp(&a.matched_len)
                .then_with(|| a_rank.cmp(b_rank))
                .then_with(|| a.path.cmp(&b.path))
        });
        Ok(ranked.into_iter().map(|(_, candidate)| candidate).collect())
    }

    fn admit_existing(
        &self,
        path: &Path,
        lease: &BlobLease,
        staged_integrity: StagedIntegrityReport,
        staging_cleanup: StagingCleanupReport,
    ) -> StoreResult<P, Option<PublishReport>> {
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
        let before = scan_managed_blobs(&self.namespace.blobs_root(), is_managed_blob_name::<P>)?;
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

    /// The store's private namespace, for tests.
    #[cfg(test)]
    pub(crate) fn namespace(&self) -> &StoreNamespace {
        &self.namespace
    }
}

/// Store internals the payload modules' tests exercise directly.
#[cfg(test)]
impl<P: DurablePayload> DurableStore<P> {
    pub(crate) fn blob_dir(&self, compatibility_id: &[u8; 32]) -> PathBuf {
        self.namespace.blob_dir(compatibility_id)
    }

    pub(crate) fn blobs_root(&self) -> PathBuf {
        self.namespace.blobs_root()
    }

    pub(crate) fn namespace_root(&self) -> PathBuf {
        self.namespace.namespace_root()
    }

    pub(crate) fn ensure_blob_dir(&self, blob_dir: &Path) -> StoreResult<P, StagingCleanupReport> {
        Ok(self.namespace.ensure_blob_dir(blob_dir)?)
    }

    pub(crate) fn lock_exclusive(&self) -> StoreResult<P, crate::checkpoint_fs::StoreLock> {
        Ok(self.namespace.lock_exclusive()?)
    }

    pub(crate) fn open_candidate(&self, path: &Path) -> StoreResult<P, Option<BlobLease>> {
        Ok(self.namespace.open_candidate(path)?)
    }
}

#[cfg(test)]
pub(crate) fn is_managed_blob_name_for<P: DurablePayload>(name: &OsStr) -> bool {
    is_managed_blob_name::<P>(name)
}

fn is_managed_blob_name<P: DurablePayload>(name: &OsStr) -> bool {
    P::parse_blob_name(name).is_some()
}
