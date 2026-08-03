//! Immutable single-file publication for explicit DeepSeek V4 checkpoints.
//!
//! The parent directory is caller-owned and trusted. Final basenames and staging
//! entries reject symlinks, but this does not defend every ancestor lookup from
//! a malicious process mutating the directory tree concurrently. Publication
//! uses a create-only hard link, so a concurrent reader or interrupted cleanup
//! may observe an additional name for the same inode; link count is not an
//! integrity signal. The private-file contract and complete record digest remain
//! authoritative.

use super::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4SnapshotFileOutcome {
    Published,
    /// The existing record has the same model, ABI, prefix, and causal digest;
    /// source-observation provenance may differ because restore never uses it.
    AlreadyPresent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SnapshotFileReport {
    pub outcome: DeepSeekV4SnapshotFileOutcome,
    pub record_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4SnapshotFileError {
    #[error("DeepSeek V4 snapshot file I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Codec(#[from] DeepSeekV4SnapshotCodecError),
    #[error("DeepSeek V4 snapshot path has no final filename")]
    MissingFilename,
    #[error("DeepSeek V4 snapshot parent is not a directory")]
    ParentNotDirectory,
    #[error("DeepSeek V4 snapshot file is not a private regular file")]
    InvalidFileMetadata,
    #[error("DeepSeek V4 snapshot destination already contains different causal state")]
    ExistingConflict,
    #[error("DeepSeek V4 staged snapshot differs after durable re-read")]
    StagedValidation,
}

pub fn publish_causal_snapshot_file(
    path: &Path,
    snapshot: &DeepSeekV4CausalSnapshot,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<DeepSeekV4SnapshotFileReport, DeepSeekV4SnapshotFileError> {
    publish_causal_snapshot_file_with_hook(path, snapshot, constraints, || {})
}

fn publish_causal_snapshot_file_with_hook(
    path: &Path,
    snapshot: &DeepSeekV4CausalSnapshot,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
    after_final_link: impl FnOnce(),
) -> Result<DeepSeekV4SnapshotFileReport, DeepSeekV4SnapshotFileError> {
    let parent = checked_parent(path)?;
    let temp_path = unique_temp_path(path, snapshot.causal_digest())?;
    let result = publish_staged(
        path,
        &temp_path,
        parent,
        snapshot,
        constraints,
        after_final_link,
    );
    let _ = std::fs::remove_file(&temp_path);
    result
}

pub fn load_causal_snapshot_file(
    path: &Path,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<DeepSeekV4CausalSnapshot, DeepSeekV4SnapshotFileError> {
    load_causal_snapshot_file_with_len(path, constraints).map(|(snapshot, _)| snapshot)
}

fn load_causal_snapshot_file_with_len(
    path: &Path,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
) -> Result<(DeepSeekV4CausalSnapshot, u64), DeepSeekV4SnapshotFileError> {
    let mut options = OpenOptions::new();
    options.read(true).custom_flags(libc::O_NOFOLLOW);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    validate_private_regular_file(&metadata, constraints.max_record_bytes)?;
    let snapshot = decode_causal_snapshot(&mut &file, constraints)?;
    let after = file.metadata()?;
    validate_private_regular_file(&after, constraints.max_record_bytes)?;
    if file_stamp(&metadata) != file_stamp(&after) {
        return Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata);
    }
    Ok((snapshot, after.len()))
}

fn publish_staged(
    final_path: &Path,
    temp_path: &Path,
    parent: &Path,
    snapshot: &DeepSeekV4CausalSnapshot,
    constraints: DeepSeekV4SnapshotCodecConstraints<'_>,
    after_final_link: impl FnOnce(),
) -> Result<DeepSeekV4SnapshotFileReport, DeepSeekV4SnapshotFileError> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW);
    let mut staged = options.open(temp_path)?;
    validate_private_regular_file(&staged.metadata()?, constraints.max_record_bytes)?;
    let encoded = {
        let mut writer = BufWriter::new(&mut staged);
        let encoded = encode_causal_snapshot(&mut writer, snapshot, constraints)?;
        writer.flush()?;
        encoded
    };
    staged.sync_all()?;
    let synced = staged.metadata()?;
    validate_private_regular_file(&synced, constraints.max_record_bytes)?;
    if synced.len() != encoded.record_bytes {
        return Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata);
    }
    staged.seek(SeekFrom::Start(0))?;
    let reread = decode_causal_snapshot(&mut staged, constraints)?;
    if !same_causal_state(&reread, snapshot) {
        return Err(DeepSeekV4SnapshotFileError::StagedValidation);
    }
    let after_decode = staged.metadata()?;
    if file_stamp(&synced) != file_stamp(&after_decode) {
        return Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata);
    }

    match std::fs::hard_link(temp_path, final_path) {
        Ok(()) => {
            after_final_link();
            std::fs::remove_file(temp_path)?;
            sync_directory(parent)?;
            let final_metadata = metadata_nofollow(final_path)?;
            validate_private_regular_file(&final_metadata, constraints.max_record_bytes)?;
            if final_metadata.dev() != after_decode.dev()
                || final_metadata.ino() != after_decode.ino()
                || final_metadata.len() != encoded.record_bytes
            {
                return Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata);
            }
            Ok(DeepSeekV4SnapshotFileReport {
                outcome: DeepSeekV4SnapshotFileOutcome::Published,
                record_bytes: encoded.record_bytes,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let (existing, existing_bytes) =
                load_causal_snapshot_file_with_len(final_path, constraints)?;
            if !same_causal_state(&existing, snapshot) {
                return Err(DeepSeekV4SnapshotFileError::ExistingConflict);
            }
            Ok(DeepSeekV4SnapshotFileReport {
                outcome: DeepSeekV4SnapshotFileOutcome::AlreadyPresent,
                record_bytes: existing_bytes,
            })
        }
        Err(error) => Err(error.into()),
    }
}

fn same_causal_state(left: &DeepSeekV4CausalSnapshot, right: &DeepSeekV4CausalSnapshot) -> bool {
    left.model_content_id() == right.model_content_id()
        && left.compatibility_digest() == right.compatibility_digest()
        && left.next_position() == right.next_position()
        && left.prefix_tokens() == right.prefix_tokens()
        && left.causal_digest() == right.causal_digest()
}

fn checked_parent(path: &Path) -> Result<&Path, DeepSeekV4SnapshotFileError> {
    if path.file_name().is_none() {
        return Err(DeepSeekV4SnapshotFileError::MissingFilename);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !std::fs::metadata(parent)?.is_dir() {
        return Err(DeepSeekV4SnapshotFileError::ParentNotDirectory);
    }
    Ok(parent)
}

fn unique_temp_path(
    final_path: &Path,
    causal_digest: &[u8; 32],
) -> Result<PathBuf, DeepSeekV4SnapshotFileError> {
    let parent = checked_parent(final_path)?;
    let filename = final_path
        .file_name()
        .ok_or(DeepSeekV4SnapshotFileError::MissingFilename)?
        .to_string_lossy();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let digest = causal_digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(parent.join(format!(
        ".{filename}.tmp-{}-{nonce}-{counter}-{digest}",
        std::process::id()
    )))
}

fn validate_private_regular_file(
    metadata: &std::fs::Metadata,
    max_record_bytes: u64,
) -> Result<(), DeepSeekV4SnapshotFileError> {
    if !metadata.file_type().is_file()
        || metadata.nlink() == 0
        || metadata.mode() & 0o077 != 0
        || metadata.len() > max_record_bytes
    {
        return Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
}

fn file_stamp(metadata: &std::fs::Metadata) -> FileStamp {
    FileStamp {
        dev: metadata.dev(),
        ino: metadata.ino(),
        len: metadata.len(),
        mtime_sec: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
    }
}

fn metadata_nofollow(path: &Path) -> Result<std::fs::Metadata, std::io::Error> {
    path.symlink_metadata()
}

fn sync_directory(path: &Path) -> Result<(), std::io::Error> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
    options.open(path)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "qwen-dsv4-snapshot-{label}-{}-{}",
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
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

    fn tiny_config() -> DeepSeekV4Config {
        let mut config = crate::deepseek_v4::flash_0731_config_fixture();
        config.key_length = 4;
        config.value_length = 4;
        config.indexer_key_length = 2;
        config.attention_kinds = vec![AttentionKind::SlidingWindow; DEEPSEEK_V4_LAYER_COUNT];
        config
    }

    fn capacity(config: &DeepSeekV4Config) -> DeepSeekV4SessionCapacity {
        DeepSeekV4SessionCapacity::for_forward_limit(3_073, config.context_length).unwrap()
    }

    fn test_snapshot(config: &DeepSeekV4Config, seed: u16) -> DeepSeekV4CausalSnapshot {
        let model_content_id = DeepSeekV4ModelContentId::new([0x5a; 32]);
        let position = 3;
        let session_capacity = capacity(config);
        let geometry = snapshot_geometry(config, session_capacity, position).unwrap();
        let prefix_tokens = vec![35, 201, 200].into_boxed_slice();
        let raw_f16_bits = (0..geometry.raw_elements)
            .map(|index| index as u16 ^ seed)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut snapshot = DeepSeekV4CausalSnapshot {
            model_content_id,
            compatibility_digest: snapshot_compatibility_digest(model_content_id, config),
            next_position: position,
            prefix_digest: prefix_digest(&prefix_tokens),
            prefix_tokens,
            source_observation: DeepSeekV4SnapshotObservation::Available,
            raw_f16_bits,
            compressor_f32_bits: Box::new([]),
            published_f16_bits: Box::new([]),
            causal_digest: [0; 32],
        };
        snapshot.causal_digest = causal_digest(&snapshot);
        validate_snapshot(&snapshot, config, session_capacity, model_content_id).unwrap();
        snapshot
    }

    fn constraints(config: &DeepSeekV4Config) -> DeepSeekV4SnapshotCodecConstraints<'_> {
        DeepSeekV4SnapshotCodecConstraints {
            config,
            session_capacity: capacity(config),
            expected_model_content_id: DeepSeekV4ModelContentId::new([0x5a; 32]),
            max_record_bytes: 1024 * 1024,
        }
    }

    #[test]
    fn immutable_file_publish_load_and_idempotence() {
        let temp = TestDir::new("publish");
        let path = temp.0.join("prefix.ds4c");
        let config = tiny_config();
        let snapshot = test_snapshot(&config, 0x1234);
        let first = publish_causal_snapshot_file(&path, &snapshot, constraints(&config)).unwrap();
        assert_eq!(first.outcome, DeepSeekV4SnapshotFileOutcome::Published);
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o077,
            0
        );
        let loaded = load_causal_snapshot_file(&path, constraints(&config)).unwrap();
        assert_eq!(loaded, snapshot);
        let second = publish_causal_snapshot_file(&path, &snapshot, constraints(&config)).unwrap();
        assert_eq!(
            second.outcome,
            DeepSeekV4SnapshotFileOutcome::AlreadyPresent
        );
        assert_eq!(second.record_bytes, first.record_bytes);
        assert_eq!(std::fs::read_dir(&temp.0).unwrap().count(), 1);
    }

    #[test]
    fn immutable_file_rejects_conflict_corruption_and_symlink() {
        let temp = TestDir::new("reject");
        let path = temp.0.join("prefix.ds4c");
        let config = tiny_config();
        let snapshot = test_snapshot(&config, 0x1234);
        publish_causal_snapshot_file(&path, &snapshot, constraints(&config)).unwrap();
        let conflict = test_snapshot(&config, 0x5678);
        assert!(matches!(
            publish_causal_snapshot_file(&path, &conflict, constraints(&config)),
            Err(DeepSeekV4SnapshotFileError::ExistingConflict)
        ));

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            load_causal_snapshot_file(&path, constraints(&config)),
            Err(DeepSeekV4SnapshotFileError::InvalidFileMetadata)
        ));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let alias = temp.0.join("alias.ds4c");
        std::fs::hard_link(&path, &alias).unwrap();
        assert_eq!(
            load_causal_snapshot_file(&path, constraints(&config)).unwrap(),
            snapshot
        );
        std::fs::remove_file(alias).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[super::codec::PAYLOAD_OFFSET] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            load_causal_snapshot_file(&path, constraints(&config)),
            Err(DeepSeekV4SnapshotFileError::Codec(
                DeepSeekV4SnapshotCodecError::DigestMismatch
            ))
        ));

        let target = temp.0.join("target");
        std::fs::write(&target, b"not a checkpoint").unwrap();
        let link = temp.0.join("link.ds4c");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(load_causal_snapshot_file(&link, constraints(&config)).is_err());
    }

    #[test]
    fn concurrent_publishers_converge_on_one_immutable_record() {
        let temp = TestDir::new("concurrent");
        let path = temp.0.join("prefix.ds4c");
        let config = tiny_config();
        let snapshot = test_snapshot(&config, 0x1234);
        let outcomes = std::thread::scope(|scope| {
            let (linked_tx, linked_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let path_ref = &path;
            let snapshot_ref = &snapshot;
            let config_ref = &config;
            let first = scope.spawn(move || {
                publish_causal_snapshot_file_with_hook(
                    path_ref,
                    snapshot_ref,
                    constraints(config_ref),
                    || {
                        linked_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    },
                )
                .unwrap()
                .outcome
            });
            linked_rx.recv().unwrap();
            let second = publish_causal_snapshot_file(&path, &snapshot, constraints(&config))
                .unwrap()
                .outcome;
            release_tx.send(()).unwrap();
            [first.join().unwrap(), second]
        });
        assert!(outcomes.contains(&DeepSeekV4SnapshotFileOutcome::Published));
        assert!(outcomes.contains(&DeepSeekV4SnapshotFileOutcome::AlreadyPresent));
        assert_eq!(
            load_causal_snapshot_file(&path, constraints(&config)).unwrap(),
            snapshot
        );
        assert_eq!(std::fs::read_dir(&temp.0).unwrap().count(), 1);
    }
}
