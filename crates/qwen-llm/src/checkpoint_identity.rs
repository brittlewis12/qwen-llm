//! Strong, lazily cached compatibility identities for durable checkpoints.
//!
//! Complete ordered GGUF shard contents are hashed once. Later invocations may
//! reuse that content root only while fstat metadata for the exact retained
//! file descriptions matches the metadata that keyed the cached entry.
//!
//! Trust envelope: source files remain immutable for a `LoadedModel` lifetime;
//! metadata reuse targets local filesystems with stable inode and nanosecond
//! timestamp semantics; and the cache root is private and trusted. BLAKE3
//! detects accidental cache corruption, not malicious replacement. Concurrent
//! source truncation retains the loader's existing possible-SIGBUS contract.
//! `ctime` is intentionally part of the key, so renames and hardlink changes
//! may conservatively force a rehash.

use crate::gguf::GgufFile;
use crate::metal_forward::SnapshotAbi;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const METADATA_DOMAIN: &[u8] = b"qwen-checkpoint-source-metadata-v1\0";
const SHARD_DOMAIN: &[u8] = b"qwen-checkpoint-shard-content-v1\0";
const CONTENT_DOMAIN: &[u8] = b"qwen-checkpoint-ordered-content-v1\0";
const COMPATIBILITY_DOMAIN: &[u8] = b"qwen-checkpoint-compatibility-v1\0";
const PARALLEL_HASH_MIN_BYTES: usize = 1024 * 1024;
const CACHE_MAGIC: &[u8; 8] = b"QWENMID\0";
const CACHE_VERSION: u32 = 1;
const CACHE_ENTRY_BYTES: usize = 128;
const CACHE_KEY_OFFSET: usize = 16;
const CACHE_CONTENT_OFFSET: usize = 48;
const CACHE_RESERVED_OFFSET: usize = 80;
const CACHE_DIGEST_OFFSET: usize = 96;

/// Bump when a state-producing numerical organization changes compatibility.
pub const STATE_NUMERICS_ABI_VERSION: u32 = 1;
/// Raw checkpoint arenas currently use the little-endian Metal state ABI.
pub const STATE_ENCODING_ABI_VERSION: u32 = 1;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SourceStamp {
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) size: u64,
    pub(crate) mtime_sec: i64,
    pub(crate) mtime_nsec: i64,
    pub(crate) ctime_sec: i64,
    pub(crate) ctime_nsec: i64,
}

impl SourceStamp {
    pub(crate) fn size(self) -> u64 {
        self.size
    }
}

pub(crate) fn source_stamp(file: &File) -> io::Result<SourceStamp> {
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint identity source is not a regular file",
        ));
    }
    Ok(SourceStamp {
        dev: metadata.dev(),
        ino: metadata.ino(),
        size: metadata.size(),
        mtime_sec: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime_sec: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    })
}

#[derive(Clone, Debug)]
pub struct CheckpointIdentityCache {
    root: PathBuf,
}

impl CheckpointIdentityCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityCacheOutcome {
    Hit,
    ComputedAndStored,
    ComputedAndRepaired,
    ComputedUncached,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointCompatibilityReport {
    pub compatibility_id: [u8; 32],
    pub content_id: [u8; 32],
    pub outcome: IdentityCacheOutcome,
    pub bytes_hashed: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum CheckpointIdentityError {
    #[error("checkpoint identity source changed since model load at shard {shard}")]
    SourceChangedSinceLoad { shard: usize },
    #[error("checkpoint identity source changed while hashing at shard {shard}")]
    SourceChangedDuringHash { shard: usize },
    #[error(
        "checkpoint identity source length mismatch at shard {shard}: file {file_bytes}, mmap {mapped_bytes}"
    )]
    SourceLengthMismatch {
        shard: usize,
        file_bytes: u64,
        mapped_bytes: u64,
    },
    #[error("checkpoint identity byte count overflow")]
    ByteCountOverflow,
    #[error("checkpoint identity I/O: {0}")]
    Io(#[from] io::Error),
}

struct SourceView<'a> {
    file: &'a File,
    bytes: &'a [u8],
    baseline: SourceStamp,
}

#[derive(Clone, Copy)]
enum CacheRead {
    Hit([u8; 32]),
    Miss,
    Corrupt,
}

pub(crate) fn checkpoint_compatibility(
    gguf: &GgufFile,
    abi: SnapshotAbi,
    cache: &CheckpointIdentityCache,
) -> Result<CheckpointCompatibilityReport, CheckpointIdentityError> {
    let sources: Vec<_> = gguf
        .shards
        .iter()
        .map(|shard| SourceView {
            file: shard.file.as_ref(),
            bytes: shard.mmap.as_ref(),
            baseline: shard.source_stamp,
        })
        .collect();
    resolve_sources(&sources, abi, cache, || {})
}

fn resolve_sources<F>(
    sources: &[SourceView<'_>],
    abi: SnapshotAbi,
    cache: &CheckpointIdentityCache,
    after_hash: F,
) -> Result<CheckpointCompatibilityReport, CheckpointIdentityError>
where
    F: FnOnce(),
{
    resolve_sources_with_hooks(sources, abi, cache, || {}, after_hash)
}

fn resolve_sources_with_hooks<C, H>(
    sources: &[SourceView<'_>],
    abi: SnapshotAbi,
    cache: &CheckpointIdentityCache,
    after_cache_read: C,
    after_hash: H,
) -> Result<CheckpointCompatibilityReport, CheckpointIdentityError>
where
    C: FnOnce(),
    H: FnOnce(),
{
    validate_sources(sources, false)?;
    let metadata_key = metadata_key(sources);
    let cache_path = cache_path(cache.root(), &metadata_key);
    let cache_read = read_cache_entry(&cache_path, &metadata_key).unwrap_or(CacheRead::Miss);
    if let CacheRead::Hit(content_id) = cache_read {
        after_cache_read();
        validate_sources(sources, false)?;
        return Ok(CheckpointCompatibilityReport {
            compatibility_id: compose_compatibility_id(content_id, abi),
            content_id,
            outcome: IdentityCacheOutcome::Hit,
            bytes_hashed: 0,
        });
    }

    let (content_id, bytes_hashed) = hash_ordered_content(sources)?;
    after_hash();
    validate_sources(sources, true)?;
    let stored = write_cache_entry(cache.root(), &cache_path, metadata_key, content_id).is_ok();
    validate_sources(sources, true)?;
    let outcome = if stored {
        match cache_read {
            CacheRead::Corrupt => IdentityCacheOutcome::ComputedAndRepaired,
            CacheRead::Miss => IdentityCacheOutcome::ComputedAndStored,
            CacheRead::Hit(_) => unreachable!(),
        }
    } else {
        IdentityCacheOutcome::ComputedUncached
    };
    Ok(CheckpointCompatibilityReport {
        compatibility_id: compose_compatibility_id(content_id, abi),
        content_id,
        outcome,
        bytes_hashed,
    })
}

fn validate_sources(
    sources: &[SourceView<'_>],
    during_hash: bool,
) -> Result<(), CheckpointIdentityError> {
    for (shard, source) in sources.iter().enumerate() {
        let current = source_stamp(source.file)?;
        if current.size != source.bytes.len() as u64 {
            return Err(CheckpointIdentityError::SourceLengthMismatch {
                shard,
                file_bytes: current.size,
                mapped_bytes: source.bytes.len() as u64,
            });
        }
        if current != source.baseline {
            return Err(if during_hash {
                CheckpointIdentityError::SourceChangedDuringHash { shard }
            } else {
                CheckpointIdentityError::SourceChangedSinceLoad { shard }
            });
        }
    }
    Ok(())
}

fn metadata_key(sources: &[SourceView<'_>]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(METADATA_DOMAIN);
    hash_u64(&mut hasher, sources.len() as u64);
    for (index, source) in sources.iter().enumerate() {
        hash_u64(&mut hasher, index as u64);
        hash_source_stamp(&mut hasher, source.baseline);
    }
    *hasher.finalize().as_bytes()
}

fn hash_ordered_content(
    sources: &[SourceView<'_>],
) -> Result<([u8; 32], u64), CheckpointIdentityError> {
    let mut content = blake3::Hasher::new();
    content.update(CONTENT_DOMAIN);
    hash_u64(&mut content, sources.len() as u64);
    let mut bytes_hashed = 0u64;
    for (index, source) in sources.iter().enumerate() {
        let mut shard = blake3::Hasher::new();
        shard.update(SHARD_DOMAIN);
        hash_u64(&mut shard, index as u64);
        hash_u64(&mut shard, source.bytes.len() as u64);
        update_content_bytes(&mut shard, source.bytes);
        let shard_id = shard.finalize();
        hash_u64(&mut content, index as u64);
        hash_u64(&mut content, source.bytes.len() as u64);
        content.update(shard_id.as_bytes());
        bytes_hashed = bytes_hashed
            .checked_add(source.bytes.len() as u64)
            .ok_or(CheckpointIdentityError::ByteCountOverflow)?;
    }
    Ok((*content.finalize().as_bytes(), bytes_hashed))
}

fn update_content_bytes<'a>(
    hasher: &'a mut blake3::Hasher,
    bytes: &[u8],
) -> &'a mut blake3::Hasher {
    if bytes.len() >= PARALLEL_HASH_MIN_BYTES {
        hasher.update_rayon(bytes)
    } else {
        hasher.update(bytes)
    }
}

fn compose_compatibility_id(content_id: [u8; 32], abi: SnapshotAbi) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COMPATIBILITY_DOMAIN);
    hasher.update(&content_id);
    hash_u32(&mut hasher, STATE_NUMERICS_ABI_VERSION);
    hash_u32(&mut hasher, STATE_ENCODING_ABI_VERSION);
    hash_u32(&mut hasher, abi.layout_version);
    hash_u32(&mut hasher, abi.n_attn_layers);
    hash_u32(&mut hasher, abi.n_gdn_layers);
    hash_u32(&mut hasher, abi.kv_dim_elements);
    hash_u32(&mut hasher, abi.kv_bytes_per_token);
    hash_u32(&mut hasher, abi.kv_storage_kind as u32);
    hash_u32(&mut hasher, abi.gdn_state_elements_per_layer);
    hash_u32(&mut hasher, abi.gdn_conv_elements_per_layer);
    *hasher.finalize().as_bytes()
}

fn cache_path(root: &Path, metadata_key: &[u8; 32]) -> PathBuf {
    root.join(format!("{}.mid", hex(metadata_key)))
}

fn read_cache_entry(path: &Path, expected_key: &[u8; 32]) -> io::Result<CacheRead> {
    let mut options = OpenOptions::new();
    options.read(true).custom_flags(libc::O_NOFOLLOW);
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(CacheRead::Miss),
        Err(error) => return Err(error),
    };
    if file.metadata()?.len() != CACHE_ENTRY_BYTES as u64 {
        return Ok(CacheRead::Corrupt);
    }
    let mut entry = [0u8; CACHE_ENTRY_BYTES];
    if file.read_exact(&mut entry).is_err() {
        return Ok(CacheRead::Corrupt);
    }
    if &entry[..8] != CACHE_MAGIC
        || get_u32(&entry, 8) != CACHE_VERSION
        || get_u32(&entry, 12) != CACHE_ENTRY_BYTES as u32
        || &entry[CACHE_KEY_OFFSET..CACHE_CONTENT_OFFSET] != expected_key
        || entry[CACHE_RESERVED_OFFSET..CACHE_DIGEST_OFFSET]
            .iter()
            .any(|&byte| byte != 0)
    {
        return Ok(CacheRead::Corrupt);
    }
    let expected_digest = blake3::hash(&entry[..CACHE_DIGEST_OFFSET]);
    if entry[CACHE_DIGEST_OFFSET..] != *expected_digest.as_bytes() {
        return Ok(CacheRead::Corrupt);
    }
    let mut content_id = [0u8; 32];
    content_id.copy_from_slice(&entry[CACHE_CONTENT_OFFSET..CACHE_RESERVED_OFFSET]);
    Ok(CacheRead::Hit(content_id))
}

fn write_cache_entry(
    root: &Path,
    final_path: &Path,
    metadata_key: [u8; 32],
    content_id: [u8; 32],
) -> io::Result<()> {
    std::fs::create_dir_all(root)?;
    let mut entry = [0u8; CACHE_ENTRY_BYTES];
    entry[..8].copy_from_slice(CACHE_MAGIC);
    put_u32(&mut entry, 8, CACHE_VERSION);
    put_u32(&mut entry, 12, CACHE_ENTRY_BYTES as u32);
    entry[CACHE_KEY_OFFSET..CACHE_CONTENT_OFFSET].copy_from_slice(&metadata_key);
    entry[CACHE_CONTENT_OFFSET..CACHE_RESERVED_OFFSET].copy_from_slice(&content_id);
    let digest = blake3::hash(&entry[..CACHE_DIGEST_OFFSET]);
    entry[CACHE_DIGEST_OFFSET..].copy_from_slice(digest.as_bytes());

    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = root.join(format!(
        ".tmp-{}-{nonce}-{sequence}-{}",
        std::process::id(),
        &hex(&metadata_key)[..16]
    ));
    let result = (|| {
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW);
        let mut file = options.open(&temp_path)?;
        file.write_all(&entry)?;
        file.sync_all()?;
        std::fs::rename(&temp_path, final_path)?;
        File::open(root)?.sync_all()?;
        match read_cache_entry(final_path, &metadata_key)? {
            CacheRead::Hit(actual) if actual == content_id => Ok(()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "published checkpoint identity cache entry did not verify",
            )),
        }
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

fn hash_source_stamp(hasher: &mut blake3::Hasher, stamp: SourceStamp) {
    hash_u64(hasher, stamp.dev);
    hash_u64(hasher, stamp.ino);
    hash_u64(hasher, stamp.size);
    hash_i64(hasher, stamp.mtime_sec);
    hash_i64(hasher, stamp.mtime_nsec);
    hash_i64(hasher, stamp.ctime_sec);
    hash_i64(hasher, stamp.ctime_nsec);
}

fn hash_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_i64(hasher: &mut blake3::Hasher, value: i64) {
    hasher.update(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("cache field"))
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
    use crate::metal_forward::{SNAPSHOT_LAYOUT_VERSION, SnapshotKvStorageKind};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qwen-checkpoint-identity-{label}-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct TestSource {
        file: File,
        mmap: memmap2::Mmap,
        baseline: SourceStamp,
    }

    impl TestSource {
        fn create(root: &Path, name: &str, bytes: &[u8]) -> Self {
            let path = root.join(name);
            std::fs::write(&path, bytes).expect("write source");
            Self::open(&path)
        }

        fn open(path: &Path) -> Self {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .expect("open source");
            let baseline = source_stamp(&file).expect("source stamp");
            let mmap = unsafe { memmap2::Mmap::map(&file).expect("map source") };
            Self {
                file,
                mmap,
                baseline,
            }
        }

        fn view(&self) -> SourceView<'_> {
            SourceView {
                file: &self.file,
                bytes: &self.mmap,
                baseline: self.baseline,
            }
        }
    }

    fn abi() -> SnapshotAbi {
        SnapshotAbi {
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 3,
            kv_dim_elements: 512,
            kv_bytes_per_token: 1024,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 16_384,
            gdn_conv_elements_per_layer: 1_024,
        }
    }

    fn restore_mtime(file: &File, stamp: SourceStamp) {
        let times = [
            libc::timespec {
                tv_sec: stamp.mtime_sec,
                tv_nsec: stamp.mtime_nsec as _,
            },
            libc::timespec {
                tv_sec: stamp.mtime_sec,
                tv_nsec: stamp.mtime_nsec as _,
            },
        ];
        let result = unsafe { libc::futimens(file.as_raw_fd(), times.as_ptr()) };
        assert_eq!(result, 0, "restore source mtime");
    }

    #[test]
    fn parallel_content_update_matches_serial_digest() {
        let bytes = (0..PARALLEL_HASH_MIN_BYTES * 2)
            .map(|index| (index.wrapping_mul(131) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut serial = blake3::Hasher::new();
        serial.update(b"framing");
        serial.update(&bytes);
        let mut parallel = blake3::Hasher::new();
        parallel.update(b"framing");
        update_content_bytes(&mut parallel, &bytes);
        assert_eq!(parallel.finalize(), serial.finalize());
    }

    #[test]
    fn compatibility_cache_hashes_once_and_recomposes_abi() {
        let temp = TestDir::new("hit");
        let source = TestSource::create(&temp.0, "model.gguf", b"complete model bytes");
        let sources = [source.view()];
        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));

        let first = resolve_sources(&sources, abi(), &cache, || {}).expect("first identity");
        assert_eq!(
            hex(&first.content_id),
            "c4b9119e7bd94aa53a360ece81503b25ae25b6f983b407e2d089ef7300cc1437"
        );
        assert_eq!(
            hex(&first.compatibility_id),
            "f2af3346d413387f84139012ea27bf81260f4f3ed4b956611d5d81828e541e5d"
        );
        assert_eq!(first.outcome, IdentityCacheOutcome::ComputedAndStored);
        assert_eq!(first.bytes_hashed, source.mmap.len() as u64);

        let second = resolve_sources(&sources, abi(), &cache, || {}).expect("cached identity");
        assert_eq!(second.outcome, IdentityCacheOutcome::Hit);
        assert_eq!(second.bytes_hashed, 0);
        assert_eq!(
            second,
            CheckpointCompatibilityReport {
                outcome: IdentityCacheOutcome::Hit,
                bytes_hashed: 0,
                ..first
            }
        );

        let mut changed_abi = abi();
        changed_abi.layout_version += 1;
        let recomposed = resolve_sources(&sources, changed_abi, &cache, || {}).unwrap();
        assert_eq!(recomposed.outcome, IdentityCacheOutcome::Hit);
        assert_eq!(recomposed.bytes_hashed, 0);
        assert_eq!(recomposed.content_id, first.content_id);
        assert_ne!(recomposed.compatibility_id, first.compatibility_id);
    }

    #[test]
    fn compatibility_content_root_binds_order_and_boundaries() {
        let temp = TestDir::new("order");
        let first = TestSource::create(&temp.0, "a.gguf", b"ab");
        let second = TestSource::create(&temp.0, "b.gguf", b"c");
        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));
        let forward =
            resolve_sources(&[first.view(), second.view()], abi(), &cache, || {}).unwrap();
        let reverse =
            resolve_sources(&[second.view(), first.view()], abi(), &cache, || {}).unwrap();
        assert_ne!(forward.content_id, reverse.content_id);

        let joined = TestSource::create(&temp.0, "joined.gguf", b"abc");
        let joined = resolve_sources(&[joined.view()], abi(), &cache, || {}).unwrap();
        assert_ne!(forward.content_id, joined.content_id);
    }

    #[test]
    fn compatibility_cache_repairs_corruption() {
        let temp = TestDir::new("repair");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");
        let sources = [source.view()];
        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));
        let first = resolve_sources(&sources, abi(), &cache, || {}).unwrap();
        let key = metadata_key(&sources);
        std::fs::write(cache_path(cache.root(), &key), [0xa5; CACHE_ENTRY_BYTES]).unwrap();

        let repaired = resolve_sources(&sources, abi(), &cache, || {}).unwrap();
        assert_eq!(repaired.outcome, IdentityCacheOutcome::ComputedAndRepaired);
        assert_eq!(repaired.content_id, first.content_id);
        let hit = resolve_sources(&sources, abi(), &cache, || {}).unwrap();
        assert_eq!(hit.outcome, IdentityCacheOutcome::Hit);
    }

    #[test]
    fn compatibility_rejects_source_mutation_before_and_during_hash() {
        let temp = TestDir::new("mutation");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");
        std::thread::sleep(std::time::Duration::from_millis(2));
        source.file.write_all_at(b"M", 0).unwrap();
        restore_mtime(&source.file, source.baseline);
        source.file.sync_all().unwrap();
        let changed = source_stamp(&source.file).unwrap();
        assert_eq!(changed.mtime_sec, source.baseline.mtime_sec);
        assert_eq!(changed.mtime_nsec, source.baseline.mtime_nsec);
        assert_ne!(
            (changed.ctime_sec, changed.ctime_nsec),
            (source.baseline.ctime_sec, source.baseline.ctime_nsec)
        );
        assert!(matches!(
            resolve_sources(
                &[source.view()],
                abi(),
                &CheckpointIdentityCache::new(temp.0.join("before")),
                || {}
            ),
            Err(CheckpointIdentityError::SourceChangedSinceLoad { shard: 0 })
        ));

        let source = TestSource::create(&temp.0, "during.gguf", b"model");
        let mutator = source.file.try_clone().unwrap();
        assert!(matches!(
            resolve_sources(
                &[source.view()],
                abi(),
                &CheckpointIdentityCache::new(temp.0.join("during")),
                move || {
                    mutator.write_all_at(b"M", 0).unwrap();
                    mutator.sync_all().unwrap();
                }
            ),
            Err(CheckpointIdentityError::SourceChangedDuringHash { shard: 0 })
        ));
    }

    #[test]
    fn compatibility_rechecks_sources_after_cache_read() {
        let temp = TestDir::new("hit-mutation");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");
        let sources = [source.view()];
        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));
        resolve_sources(&sources, abi(), &cache, || {}).unwrap();

        let mutator = source.file.try_clone().unwrap();
        assert!(matches!(
            resolve_sources_with_hooks(
                &sources,
                abi(),
                &cache,
                move || {
                    mutator.write_all_at(b"M", 0).unwrap();
                    mutator.sync_all().unwrap();
                },
                || {}
            ),
            Err(CheckpointIdentityError::SourceChangedSinceLoad { shard: 0 })
        ));
    }

    #[test]
    fn compatibility_uses_retained_descriptor_after_symlink_retarget() {
        let temp = TestDir::new("symlink");
        let first_path = temp.0.join("first.gguf");
        let second_path = temp.0.join("second.gguf");
        std::fs::write(&first_path, b"first").unwrap();
        std::fs::write(&second_path, b"other").unwrap();
        let link = temp.0.join("model.gguf");
        std::os::unix::fs::symlink(&first_path, &link).unwrap();
        let source = TestSource::open(&link);

        let replacement = temp.0.join("replacement-link");
        std::os::unix::fs::symlink(&second_path, &replacement).unwrap();
        std::fs::rename(replacement, &link).unwrap();

        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));
        let retained = resolve_sources(&[source.view()], abi(), &cache, || {}).unwrap();
        let second = TestSource::open(&second_path);
        let retargeted = resolve_sources(&[second.view()], abi(), &cache, || {}).unwrap();
        assert_ne!(retained.content_id, retargeted.content_id);
        assert_eq!(retained.bytes_hashed, 5);
    }

    #[test]
    fn compatibility_survives_cache_write_failure_without_weakening_id() {
        let temp = TestDir::new("uncached");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");
        let blocked_root = temp.0.join("not-a-directory");
        std::fs::write(&blocked_root, b"x").unwrap();
        let report = resolve_sources(
            &[source.view()],
            abi(),
            &CheckpointIdentityCache::new(blocked_root),
            || {},
        )
        .unwrap();
        assert_eq!(report.outcome, IdentityCacheOutcome::ComputedUncached);
        assert_eq!(report.bytes_hashed, source.mmap.len() as u64);
    }
}
