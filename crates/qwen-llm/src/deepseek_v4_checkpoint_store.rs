//! DeepSeek V4 causal-snapshot checkpoints on the generic durable store
//! ([`crate::durable_store`]): record naming, prefix keys, codec error
//! classification, and the session-typed restore/prepare helpers. The store
//! itself (publication, lookup, budgets, locking) is shared with Qwen.

use crate::checkpoint_fs::{hex, parse_hex_32};
use crate::checkpoint_store::PublishReport;
use crate::deepseek_v4::DeepSeekV4Config;
use crate::deepseek_v4_metal::{
    DeepSeekV4CausalSnapshot, DeepSeekV4CompatibilityDigest, DeepSeekV4MetalError,
    DeepSeekV4ModelContentId, DeepSeekV4Session, DeepSeekV4SessionCapacity,
    DeepSeekV4SnapshotCodecConstraints, DeepSeekV4SnapshotCodecError, decode_causal_snapshot,
    encode_causal_snapshot, encoded_causal_snapshot_record_bytes,
};
use crate::durable_store::{
    DurablePayload, DurableStore, DurableStoreError, LookupReport, LookupVerdict,
};
use std::collections::{BTreeSet, HashMap};
use std::io::{self, Read, Write};

const NAMESPACE_VERSION: &str = "dsv4-v1";
const PREFIX_KEY_DOMAIN: &[u8] = b"qwen-dsv4-checkpoint-prefix-key-v1\0";
const BLOB_EXTENSION: &str = "dsv4cp";

pub type DeepSeekV4CheckpointStore = DurableStore<DeepSeekV4CausalSnapshot>;
pub type DeepSeekV4CheckpointStoreError = DurableStoreError<DeepSeekV4SnapshotCodecError>;
pub type DeepSeekV4PublishReport = PublishReport;
pub type DeepSeekV4LookupReport = LookupReport<DeepSeekV4CausalSnapshot>;

impl From<DeepSeekV4SnapshotCodecError> for DeepSeekV4CheckpointStoreError {
    fn from(error: DeepSeekV4SnapshotCodecError) -> Self {
        Self::Codec(error)
    }
}

impl DeepSeekV4CheckpointStore {
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
}

impl DurablePayload for DeepSeekV4CausalSnapshot {
    type Token = u32;
    type Context<'a> = DeepSeekV4StoreContext<'a>;
    type CodecError = DeepSeekV4SnapshotCodecError;
    /// One record per prefix: no mode.
    type Mode = ();
    const FAMILY_LABEL: &'static str = "DeepSeek V4";
    const NAMESPACE_VERSION: &'static str = NAMESPACE_VERSION;
    /// Causal state is a pure function of its prefix: a different valid
    /// record under the same key is a collision, never a repair.
    const MISMATCHED_EXISTING_IS_COLLISION: bool = true;

    fn compatibility_id(context: &DeepSeekV4StoreContext<'_>) -> [u8; 32] {
        *context.compatibility_digest.as_bytes()
    }

    fn check_publish(
        &self,
        context: &DeepSeekV4StoreContext<'_>,
    ) -> Result<(), DeepSeekV4CheckpointStoreError> {
        context.validate()?;
        if self.compatibility_digest() != context.compatibility_digest {
            return Err(compatibility_mismatch());
        }
        Ok(())
    }

    fn check_lookup(
        context: &DeepSeekV4StoreContext<'_>,
    ) -> Result<(), DeepSeekV4CheckpointStoreError> {
        context.validate()
    }

    fn matched_len(&self) -> usize {
        self.next_position() as usize
    }

    fn mode(&self) {}

    fn prefix_key(&self, compatibility_id: &[u8; 32]) -> [u8; 32] {
        snapshot_prefix_key(compatibility_id, self)
    }

    fn request_prefix_keys(
        compatibility_id: &[u8; 32],
        request_tokens: &[u32],
        lengths: &BTreeSet<usize>,
    ) -> HashMap<usize, [u8; 32]> {
        request_prefix_keys(compatibility_id, request_tokens, lengths)
    }

    fn blob_name(matched_len: usize, _mode: (), digest: &[u8; 32]) -> String {
        blob_name(matched_len, digest)
    }

    fn parse_blob_name(name: &std::ffi::OsStr) -> Option<(usize, (), [u8; 32])> {
        parse_blob_name(name)
    }

    /// A snapshot carries no observation, so it must leave at least one
    /// request token to prefill.
    fn candidate_rank(_mode: (), covers_request: bool) -> Option<usize> {
        (!covers_request).then_some(0)
    }

    fn restored_len(_mode: (), matched_len: usize) -> usize {
        matched_len
    }

    fn lookup_verdict(
        &self,
        request_tokens: &[u32],
        matched_len: usize,
        _mode: (),
        context: &DeepSeekV4StoreContext<'_>,
        digest: &[u8; 32],
    ) -> LookupVerdict {
        if namespace_matches(
            self,
            request_tokens,
            matched_len,
            context.compatibility_digest.as_bytes(),
            digest,
        ) {
            LookupVerdict::Match
        } else if self.next_position() as usize == matched_len
            && same_request_prefix(self, request_tokens)
            && self.compatibility_digest() == context.compatibility_digest
        {
            LookupVerdict::Collision
        } else {
            LookupVerdict::Mismatch
        }
    }

    fn same_checkpoint(&self, other: &Self) -> bool {
        same_causal_state(self, other)
    }

    fn encoded_record_bytes(
        &self,
        context: DeepSeekV4StoreContext<'_>,
    ) -> Result<u64, DeepSeekV4SnapshotCodecError> {
        // Preflight sizes against the inner codec budget, as before the
        // generic store; encode/decode use the outer `max_record_bytes`.
        // Every caller sets both equal.
        encoded_causal_snapshot_record_bytes(self, context.codec_constraints)
    }

    fn encode<W: Write>(
        &self,
        writer: &mut W,
        context: DeepSeekV4StoreContext<'_>,
    ) -> Result<u64, DeepSeekV4SnapshotCodecError> {
        Ok(encode_causal_snapshot(writer, self, context.codec_constraints())?.record_bytes)
    }

    fn decode<R: Read>(
        reader: &mut R,
        context: DeepSeekV4StoreContext<'_>,
    ) -> Result<Self, DeepSeekV4SnapshotCodecError> {
        decode_causal_snapshot(reader, context.codec_constraints())
    }

    fn codec_error_proves_invalid(error: &DeepSeekV4SnapshotCodecError) -> bool {
        codec_error_proves_invalid_blob(error)
    }

    fn codec_error_is_io(error: &DeepSeekV4SnapshotCodecError) -> bool {
        matches!(error, DeepSeekV4SnapshotCodecError::Io(_))
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
            return Err(compatibility_mismatch());
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

fn compatibility_mismatch() -> DeepSeekV4CheckpointStoreError {
    DeepSeekV4CheckpointStoreError::CompatibilityMismatch {
        family: <DeepSeekV4CausalSnapshot as DurablePayload>::FAMILY_LABEL,
    }
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

fn parse_blob_name(name: &std::ffi::OsStr) -> Option<(usize, (), [u8; 32])> {
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
    Some((matched_len, (), digest))
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
    use crate::checkpoint_fs::scan_managed_blobs;
    use crate::durable_store::PublishOutcome;
    use std::path::{Path, PathBuf};

    fn is_managed_blob_name(name: &std::ffi::OsStr) -> bool {
        crate::durable_store::is_managed_blob_name_for::<DeepSeekV4CausalSnapshot>(name)
    }
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
        assert_eq!(parsed.0, 12);
        assert_eq!(parsed.2, digest);
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
            .namespace()
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

        // Over this process's record budget: skipped and kept, not an error.
        let report = store
            .lookup(fixture.context(published.blob_bytes - 1), &extension)
            .unwrap();
        assert!(report.snapshot.is_none());
        assert_eq!(report.unusable_skipped, 1);
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
            Err(DeepSeekV4CheckpointStoreError::NamespaceCollision { .. })
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
