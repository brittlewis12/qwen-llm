//! Strong, lazily cached compatibility identities for durable checkpoints.
//!
//! Complete ordered GGUF shard contents are hashed once. Later invocations may
//! reuse that content root only while fstat metadata for the exact retained
//! file descriptions matches the metadata that keyed the cached entry.
//!
//! When every shard carries a fresh Hugging Face download sidecar
//! (`.cache/huggingface/download/<name>.metadata`, whose LFS etag line is the
//! shard's SHA-256), the content root may instead be composed from those
//! declared digests under separate domains, skipping the full read. Declared
//! and hashed roots are distinct identities; whichever resolves first for a
//! given source description is cached and reused. `QWEN_CHECKPOINT_MODEL_IDENTITY=hashed`
//! forces the exhaustive read.
//!
//! Trust envelope: source files remain immutable for a `LoadedModel` lifetime;
//! metadata reuse targets local filesystems with stable inode and nanosecond
//! timestamp semantics; and the cache root is private and trusted. BLAKE3
//! detects accidental cache corruption, not malicious replacement. Declared
//! digests additionally trust the downloader's verification and the source
//! mtime/ctime ordering against the sidecar timestamp. Concurrent source truncation
//! retains the loader's existing possible-SIGBUS contract. `ctime` is
//! intentionally part of the key, so renames and hardlink changes may
//! conservatively force a rehash.

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
const DECLARED_SHARD_DOMAIN: &[u8] = b"qwen-checkpoint-declared-shard-sha256-v1\0";
const DECLARED_CONTENT_DOMAIN: &[u8] = b"qwen-checkpoint-declared-ordered-content-v1\0";
const COMPATIBILITY_DOMAIN: &[u8] = b"qwen-checkpoint-compatibility-v1\0";
const IDENTITY_MODE_ENV: &str = "QWEN_CHECKPOINT_MODEL_IDENTITY";
const SIDECAR_MAX_BYTES: u64 = 4096;
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
    /// Composed from fresh downloader-declared shard digests; nothing read.
    DeclaredAndStored,
    /// Declared digests resolved but the cache entry could not be written.
    DeclaredUncached,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointCompatibilityReport {
    pub compatibility_id: [u8; 32],
    pub content_id: [u8; 32],
    pub outcome: IdentityCacheOutcome,
    pub bytes_hashed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointContentReport {
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
    #[error(
        "{IDENTITY_MODE_ENV}={0:?} is not a recognized checkpoint identity mode; use auto or hashed"
    )]
    IdentityModeEnv(String),
    #[error(
        "checkpoint content identity has no matching cache entry or fresh declared shard digests; refusing to hash model bytes"
    )]
    HashingRequired,
    #[error("checkpoint identity I/O: {0}")]
    Io(#[from] io::Error),
}

struct SourceView<'a> {
    file: &'a File,
    path: &'a Path,
    bytes: &'a [u8],
    baseline: SourceStamp,
}

/// How a cold content root may be derived. Cached roots are reused verbatim
/// regardless of mode; the mode only governs the first resolution for a given
/// source description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdentityMode {
    /// Compose from fresh downloader-declared SHA-256 sidecars when every
    /// shard has one; otherwise hash the ordered shard contents.
    Auto,
    /// Always hash the ordered shard contents.
    Hashed,
}

fn configured_identity_mode() -> Result<IdentityMode, CheckpointIdentityError> {
    match std::env::var(IDENTITY_MODE_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(IdentityMode::Auto),
        Err(std::env::VarError::NotUnicode(_)) => Err(CheckpointIdentityError::IdentityModeEnv(
            "<non-unicode>".to_string(),
        )),
        Ok(value) => match value.as_str() {
            "auto" => Ok(IdentityMode::Auto),
            "hashed" => Ok(IdentityMode::Hashed),
            other => Err(CheckpointIdentityError::IdentityModeEnv(other.to_string())),
        },
    }
}

#[derive(Clone, Copy)]
enum CacheRead {
    Hit([u8; 32]),
    Miss,
    Corrupt,
}

pub fn checkpoint_content_identity(
    gguf: &GgufFile,
    cache: &CheckpointIdentityCache,
) -> Result<CheckpointContentReport, CheckpointIdentityError> {
    let sources = source_views(gguf);
    resolve_content_sources(&sources, cache, || {})
}

/// Resolve a strong cached or downloader-declared content root without ever
/// hashing mapped model bytes. A cold source without complete fresh sidecars
/// fails closed instead of falling back to an exhaustive scan.
pub fn checkpoint_content_identity_without_weight_hashing(
    gguf: &GgufFile,
    cache: &CheckpointIdentityCache,
) -> Result<CheckpointContentReport, CheckpointIdentityError> {
    let sources = source_views(gguf);
    resolve_content_sources_without_weight_hashing(&sources, cache)
}

/// Report whether every shard carries a fresh declared-digest sidecar, without
/// resolving or caching an identity.
pub fn declared_identity_available(gguf: &GgufFile) -> bool {
    let sources = source_views(gguf);
    declared_shard_digests(&sources).is_some()
}

pub(crate) fn checkpoint_compatibility(
    gguf: &GgufFile,
    abi: SnapshotAbi,
    cache: &CheckpointIdentityCache,
) -> Result<CheckpointCompatibilityReport, CheckpointIdentityError> {
    let sources = source_views(gguf);
    resolve_sources(&sources, abi, cache, || {})
}

fn source_views(gguf: &GgufFile) -> Vec<SourceView<'_>> {
    gguf.shards
        .iter()
        .map(|shard| SourceView {
            file: shard.file.as_ref(),
            path: shard.path.as_path(),
            bytes: shard.mmap.as_ref(),
            baseline: shard.source_stamp,
        })
        .collect()
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
    let content = resolve_content_sources_with_hooks(sources, cache, after_cache_read, after_hash)?;
    Ok(CheckpointCompatibilityReport {
        compatibility_id: compose_compatibility_id(content.content_id, abi),
        content_id: content.content_id,
        outcome: content.outcome,
        bytes_hashed: content.bytes_hashed,
    })
}

fn resolve_content_sources<H>(
    sources: &[SourceView<'_>],
    cache: &CheckpointIdentityCache,
    after_hash: H,
) -> Result<CheckpointContentReport, CheckpointIdentityError>
where
    H: FnOnce(),
{
    resolve_content_sources_with_hooks(sources, cache, || {}, after_hash)
}

fn resolve_content_sources_with_hooks<C, H>(
    sources: &[SourceView<'_>],
    cache: &CheckpointIdentityCache,
    after_cache_read: C,
    after_hash: H,
) -> Result<CheckpointContentReport, CheckpointIdentityError>
where
    C: FnOnce(),
    H: FnOnce(),
{
    let mode = configured_identity_mode()?;
    resolve_content_sources_with_mode(sources, cache, mode, after_cache_read, after_hash)
}

fn resolve_content_sources_with_mode<C, H>(
    sources: &[SourceView<'_>],
    cache: &CheckpointIdentityCache,
    mode: IdentityMode,
    after_cache_read: C,
    after_hash: H,
) -> Result<CheckpointContentReport, CheckpointIdentityError>
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
        return Ok(CheckpointContentReport {
            content_id,
            outcome: IdentityCacheOutcome::Hit,
            bytes_hashed: 0,
        });
    }

    let declared = match mode {
        IdentityMode::Auto => declared_ordered_content(sources),
        IdentityMode::Hashed => None,
    };
    let (content_id, bytes_hashed, declared_used) = match declared {
        Some(content_id) => (content_id, 0, true),
        None => {
            let (content_id, bytes_hashed) = hash_ordered_content(sources)?;
            (content_id, bytes_hashed, false)
        }
    };
    after_hash();
    validate_sources(sources, true)?;
    let stored = write_cache_entry(cache.root(), &cache_path, metadata_key, content_id).is_ok();
    validate_sources(sources, true)?;
    let outcome = match (stored, declared_used) {
        (true, true) => IdentityCacheOutcome::DeclaredAndStored,
        (false, true) => IdentityCacheOutcome::DeclaredUncached,
        (true, false) => match cache_read {
            CacheRead::Corrupt => IdentityCacheOutcome::ComputedAndRepaired,
            CacheRead::Miss => IdentityCacheOutcome::ComputedAndStored,
            CacheRead::Hit(_) => unreachable!(),
        },
        (false, false) => IdentityCacheOutcome::ComputedUncached,
    };
    Ok(CheckpointContentReport {
        content_id,
        outcome,
        bytes_hashed,
    })
}

fn resolve_content_sources_without_weight_hashing(
    sources: &[SourceView<'_>],
    cache: &CheckpointIdentityCache,
) -> Result<CheckpointContentReport, CheckpointIdentityError> {
    validate_sources(sources, false)?;
    let metadata_key = metadata_key(sources);
    let cache_path = cache_path(cache.root(), &metadata_key);
    if let CacheRead::Hit(content_id) =
        read_cache_entry(&cache_path, &metadata_key).unwrap_or(CacheRead::Miss)
    {
        validate_sources(sources, false)?;
        return Ok(CheckpointContentReport {
            content_id,
            outcome: IdentityCacheOutcome::Hit,
            bytes_hashed: 0,
        });
    }
    let content_id =
        declared_ordered_content(sources).ok_or(CheckpointIdentityError::HashingRequired)?;
    validate_sources(sources, false)?;
    let stored = write_cache_entry(cache.root(), &cache_path, metadata_key, content_id).is_ok();
    validate_sources(sources, false)?;
    Ok(CheckpointContentReport {
        content_id,
        outcome: if stored {
            IdentityCacheOutcome::DeclaredAndStored
        } else {
            IdentityCacheOutcome::DeclaredUncached
        },
        bytes_hashed: 0,
    })
}

/// Compose the declared content root when every shard has a fresh sidecar.
///
/// Any missing, malformed, oversized, or stale sidecar disqualifies the whole
/// declared derivation. Permissive callers hash the content; strict callers
/// fail closed.
fn declared_ordered_content(sources: &[SourceView<'_>]) -> Option<[u8; 32]> {
    let digests = declared_shard_digests(sources)?;
    let mut content = blake3::Hasher::new();
    content.update(DECLARED_CONTENT_DOMAIN);
    hash_u64(&mut content, sources.len() as u64);
    for (index, (source, digest)) in sources.iter().zip(&digests).enumerate() {
        let mut shard = blake3::Hasher::new();
        shard.update(DECLARED_SHARD_DOMAIN);
        hash_u64(&mut shard, index as u64);
        hash_u64(&mut shard, source.baseline.size());
        shard.update(digest);
        let shard_id = shard.finalize();
        hash_u64(&mut content, index as u64);
        hash_u64(&mut content, source.baseline.size());
        content.update(shard_id.as_bytes());
    }
    Some(*content.finalize().as_bytes())
}

fn declared_shard_digests(sources: &[SourceView<'_>]) -> Option<Vec<[u8; 32]>> {
    sources
        .iter()
        .map(|source| declared_shard_digest(source))
        .collect()
}

fn declared_shard_digest(source: &SourceView<'_>) -> Option<[u8; 32]> {
    let name = source.path.file_name()?;
    let mut sidecar_name = name.to_os_string();
    sidecar_name.push(".metadata");
    let sidecar_path = source
        .path
        .parent()?
        .join(".cache")
        .join("huggingface")
        .join("download")
        .join(sidecar_name);
    let mut options = OpenOptions::new();
    options.read(true).custom_flags(libc::O_NOFOLLOW);
    let mut file = options.open(&sidecar_path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.file_type().is_file() || metadata.len() > SIDECAR_MAX_BYTES {
        return None;
    }
    let mut raw = String::new();
    file.read_to_string(&mut raw).ok()?;
    parse_declared_sidecar(&raw, source.baseline)
}

/// Parse the three-line hf download sidecar: commit hash, etag, timestamp.
///
/// Only LFS-style 64-hex etags declare a content SHA-256; anything else is
/// not a usable declaration. The shard must not have been modified after the
/// sidecar was stamped.
fn parse_declared_sidecar(raw: &str, baseline: SourceStamp) -> Option<[u8; 32]> {
    let mut lines = raw.lines();
    let commit = lines.next()?.trim();
    let etag = lines.next()?.trim();
    let timestamp = lines.next()?.trim();
    if lines.any(|line| !line.trim().is_empty()) {
        return None;
    }
    if commit.len() != 40 || !commit.bytes().all(is_lower_hex_byte) {
        return None;
    }
    let digest = parse_hex_32(etag)?;
    let stamped: f64 = timestamp.parse().ok()?;
    if !stamped.is_finite() || stamped < 0.0 {
        return None;
    }
    let source_mtime = baseline.mtime_sec as f64 + baseline.mtime_nsec as f64 * 1e-9;
    let source_ctime = baseline.ctime_sec as f64 + baseline.ctime_nsec as f64 * 1e-9;
    if stamped < source_mtime.max(source_ctime) {
        return None;
    }
    Some(digest)
}

fn is_lower_hex_byte(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

fn parse_hex_32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(is_lower_hex_byte) {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        out[index] = hex_nibble(pair[0])? << 4 | hex_nibble(pair[1])?;
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
        path: PathBuf,
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
                path: path.to_path_buf(),
                mmap,
                baseline,
            }
        }

        fn view(&self) -> SourceView<'_> {
            SourceView {
                file: &self.file,
                path: &self.path,
                bytes: &self.mmap,
                baseline: self.baseline,
            }
        }

        fn write_sidecar(&self, commit: &str, etag: &str, timestamp: f64) {
            let sidecar_dir = self
                .path
                .parent()
                .unwrap()
                .join(".cache")
                .join("huggingface")
                .join("download");
            std::fs::create_dir_all(&sidecar_dir).unwrap();
            let mut name = self.path.file_name().unwrap().to_os_string();
            name.push(".metadata");
            std::fs::write(
                sidecar_dir.join(name),
                format!("{commit}\n{etag}\n{timestamp}\n"),
            )
            .unwrap();
        }

        fn fresh_sidecar_timestamp(&self) -> f64 {
            let mtime = self.baseline.mtime_sec as f64 + self.baseline.mtime_nsec as f64 * 1e-9;
            let ctime = self.baseline.ctime_sec as f64 + self.baseline.ctime_nsec as f64 * 1e-9;
            mtime.max(ctime) + 1.0
        }
    }

    const TEST_COMMIT: &str = "9f8c8a7abcd5f02d598f5b6c194bab804e87b837";
    const TEST_SHA256: &str = "15fc87ee87c445a1732b321673df56f9d5675aec4db46dcc9436d29b6e41f3c8";

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
    fn declared_sidecars_compose_identity_without_reading_content() {
        let temp = TestDir::new("declared");
        let source = TestSource::create(&temp.0, "model.gguf", b"declared model bytes");
        source.write_sidecar(TEST_COMMIT, TEST_SHA256, source.fresh_sidecar_timestamp());
        let sources = [source.view()];
        let declared_cache = CheckpointIdentityCache::new(temp.0.join("declared-cache"));

        let declared = resolve_content_sources(&sources, &declared_cache, || {}).unwrap();
        assert_eq!(declared.outcome, IdentityCacheOutcome::DeclaredAndStored);
        assert_eq!(declared.bytes_hashed, 0);

        let hashed_cache = CheckpointIdentityCache::new(temp.0.join("hashed-cache"));
        let hashed = resolve_content_sources_with_mode(
            &sources,
            &hashed_cache,
            IdentityMode::Hashed,
            || {},
            || {},
        )
        .unwrap();
        assert_eq!(hashed.outcome, IdentityCacheOutcome::ComputedAndStored);
        assert_eq!(hashed.bytes_hashed, source.mmap.len() as u64);
        assert_ne!(
            declared.content_id, hashed.content_id,
            "declared and hashed roots are domain-separated identities"
        );

        // The cached root is sticky: later resolutions reuse it even after
        // the sidecar disappears or the preferred mode changes.
        std::fs::remove_dir_all(temp.0.join(".cache")).unwrap();
        let sticky = resolve_content_sources(&sources, &declared_cache, || {}).unwrap();
        assert_eq!(sticky.outcome, IdentityCacheOutcome::Hit);
        assert_eq!(sticky.content_id, declared.content_id);
        let sticky_hashed_mode = resolve_content_sources_with_mode(
            &sources,
            &declared_cache,
            IdentityMode::Hashed,
            || {},
            || {},
        )
        .unwrap();
        assert_eq!(sticky_hashed_mode.outcome, IdentityCacheOutcome::Hit);
        assert_eq!(sticky_hashed_mode.content_id, declared.content_id);
    }

    #[test]
    fn strict_non_hashing_identity_uses_declarations_or_cache_only() {
        let temp = TestDir::new("strict-declared");
        let source = TestSource::create(&temp.0, "model.gguf", b"model bytes must not be hashed");
        let sources = [source.view()];
        let cache = CheckpointIdentityCache::new(temp.0.join("identity"));
        assert!(matches!(
            resolve_content_sources_without_weight_hashing(&sources, &cache),
            Err(CheckpointIdentityError::HashingRequired)
        ));

        source.write_sidecar(TEST_COMMIT, TEST_SHA256, source.fresh_sidecar_timestamp());
        let declared = resolve_content_sources_without_weight_hashing(&sources, &cache).unwrap();
        assert_eq!(declared.outcome, IdentityCacheOutcome::DeclaredAndStored);
        assert_eq!(declared.bytes_hashed, 0);

        std::fs::remove_dir_all(temp.0.join(".cache")).unwrap();
        let cached = resolve_content_sources_without_weight_hashing(&sources, &cache).unwrap();
        assert_eq!(cached.outcome, IdentityCacheOutcome::Hit);
        assert_eq!(cached.content_id, declared.content_id);
        assert_eq!(cached.bytes_hashed, 0);
    }

    #[test]
    fn declared_identity_rejects_sidecar_older_than_source_ctime() {
        let temp = TestDir::new("declared-ctime");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");
        let stamped = source.fresh_sidecar_timestamp();
        let mut replaced = source.baseline;
        replaced.ctime_sec = (stamped.floor() as i64).saturating_add(1);
        replaced.ctime_nsec = 0;
        let sidecar = format!("{TEST_COMMIT}\n{TEST_SHA256}\n{stamped}\n");
        assert!(parse_declared_sidecar(&sidecar, replaced).is_none());
    }

    #[test]
    fn declared_identity_requires_fresh_wellformed_sidecars_on_every_shard() {
        let temp = TestDir::new("declared-reject");
        let source = TestSource::create(&temp.0, "model.gguf", b"model");

        // Shard modified after the sidecar was stamped: stale declaration.
        source.write_sidecar(
            TEST_COMMIT,
            TEST_SHA256,
            source.baseline.mtime_sec as f64 - 10.0,
        );
        let stale = resolve_content_sources(
            &[source.view()],
            &CheckpointIdentityCache::new(temp.0.join("stale")),
            || {},
        )
        .unwrap();
        assert_eq!(stale.outcome, IdentityCacheOutcome::ComputedAndStored);
        assert!(stale.bytes_hashed > 0);

        // Non-LFS etag shapes (md5-style 32 hex) do not declare content.
        source.write_sidecar(
            TEST_COMMIT,
            "d41d8cd98f00b204e9800998ecf8427e",
            source.fresh_sidecar_timestamp(),
        );
        let malformed = resolve_content_sources(
            &[source.view()],
            &CheckpointIdentityCache::new(temp.0.join("malformed")),
            || {},
        )
        .unwrap();
        assert_eq!(malformed.outcome, IdentityCacheOutcome::ComputedAndStored);

        // Every shard must declare; one bare shard rejects the whole set.
        source.write_sidecar(TEST_COMMIT, TEST_SHA256, source.fresh_sidecar_timestamp());
        let bare = TestSource::create(&temp.0, "model-shard2.gguf", b"more");
        let partial = resolve_content_sources(
            &[source.view(), bare.view()],
            &CheckpointIdentityCache::new(temp.0.join("partial")),
            || {},
        )
        .unwrap();
        assert_eq!(partial.outcome, IdentityCacheOutcome::ComputedAndStored);
        assert_eq!(
            partial.bytes_hashed,
            (source.mmap.len() + bare.mmap.len()) as u64
        );
    }

    #[test]
    fn declared_root_binds_shard_order_and_declared_digests() {
        let temp = TestDir::new("declared-order");
        let first = TestSource::create(&temp.0, "a.gguf", b"aa");
        let second = TestSource::create(&temp.0, "b.gguf", b"bb");
        first.write_sidecar(TEST_COMMIT, TEST_SHA256, first.fresh_sidecar_timestamp());
        second.write_sidecar(
            TEST_COMMIT,
            "25fc87ee87c445a1732b321673df56f9d5675aec4db46dcc9436d29b6e41f3c8",
            second.fresh_sidecar_timestamp(),
        );
        let forward = declared_ordered_content(&[first.view(), second.view()]).unwrap();
        let reverse = declared_ordered_content(&[second.view(), first.view()]).unwrap();
        assert_ne!(forward, reverse);

        second.write_sidecar(TEST_COMMIT, TEST_SHA256, second.fresh_sidecar_timestamp());
        let redeclared = declared_ordered_content(&[first.view(), second.view()]).unwrap();
        assert_ne!(forward, redeclared);
    }

    #[test]
    fn declared_sidecar_parser_is_strict() {
        let stamp = SourceStamp {
            dev: 1,
            ino: 2,
            size: 3,
            mtime_sec: 1_000,
            mtime_nsec: 0,
            ctime_sec: 1_000,
            ctime_nsec: 0,
        };
        let valid = format!("{TEST_COMMIT}\n{TEST_SHA256}\n2000.5\n");
        assert!(parse_declared_sidecar(&valid, stamp).is_some());
        for rejected in [
            format!("{TEST_COMMIT}\n{TEST_SHA256}\n2000.5\ntrailing\n"),
            format!("{TEST_COMMIT}\n{}\n2000.5\n", TEST_SHA256.to_uppercase()),
            format!("{}\n{TEST_SHA256}\n2000.5\n", &TEST_COMMIT[..39]),
            format!("{TEST_COMMIT}\n{TEST_SHA256}\nNaN\n"),
            format!("{TEST_COMMIT}\n{TEST_SHA256}\n-4.0\n"),
            format!("{TEST_COMMIT}\n{TEST_SHA256}\n"),
        ] {
            assert_eq!(
                parse_declared_sidecar(&rejected, stamp),
                None,
                "{rejected:?}"
            );
        }
        let too_old = SourceStamp {
            mtime_sec: 2_010,
            ..stamp
        };
        assert_eq!(parse_declared_sidecar(&valid, too_old), None);
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
