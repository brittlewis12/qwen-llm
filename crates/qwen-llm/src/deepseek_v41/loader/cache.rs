//! Private, trusted local cache. Sources and published buffers must remain
//! immutable for the lifetime of their stores/mappings, as for checkpoint identity.
use super::*;
use crate::checkpoint_fs;
use crate::checkpoint_identity::source_stamp;
use crate::metal::{MetalContext, MetalTensor};
use crate::tensor::GgmlType;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::PathBuf;

/// Bump for any conversion arithmetic, source selection, or output ABI change.
pub const CONVERTER_LAYOUT_VERSION: &str = "dsv41-reap272-mxfp4-f32-v1";

pub(super) struct Cache {
    root: PathBuf,
    _lock: File,
}

impl Cache {
    pub(super) fn open(root: &Path) -> Result<Self, String> {
        checkpoint_fs::ensure_real_directory(root).map_err(|e| e.to_string())?;
        let metadata = std::fs::metadata(root).map_err(|e| e.to_string())?;
        if metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o077 != 0 {
            return Err("converted cache must be caller-owned and private (mode 0700)".into());
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(root.join("writer.lock"))
            .map_err(|e| e.to_string())?;
        checkpoint_fs::flock_retry(&lock, libc::LOCK_EX).map_err(|e| e.to_string())?;
        Ok(Self {
            root: root.to_owned(),
            _lock: lock,
        })
    }
}

pub(super) struct Entry<'a> {
    cache: &'a Cache,
    sources: Vec<(&'a str, &'a StoredTensor)>,
    spec: String,
    slot: String,
    bytes: u64,
    shape: Vec<u64>,
    dtype: GgmlType,
}

impl<'a> Entry<'a> {
    pub(super) fn new(
        cache: &'a Cache,
        sources: Vec<(&'a str, &'a StoredTensor)>,
        bytes: u64,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, String> {
        let mut spec = format!("{CONVERTER_LAYOUT_VERSION}\n{bytes}\n{shape:?}\n{dtype:?}\n");
        let mut stamps = String::new();
        for (name, tensor) in &sources {
            spec.push_str(&format!("{name:?}:{:?}\n", tensor.descriptor));
            stamps.push_str(&format!("{:?}\n", tensor.stamp));
        }
        let slot = blake3::hash(format!("{spec}{stamps}").as_bytes())
            .to_hex()
            .to_string();
        let entry = Self {
            cache,
            sources,
            spec,
            slot,
            bytes,
            shape,
            dtype,
        };
        entry.validate_sources()?;
        Ok(entry)
    }

    fn validate_sources(&self) -> Result<(), String> {
        let mut seen = std::collections::BTreeSet::new();
        for (_, tensor) in &self.sources {
            if seen.insert((tensor.stamp.dev, tensor.stamp.ino))
                && source_stamp(&tensor.file).map_err(|e| e.to_string())? != tensor.stamp
            {
                return Err("source mutated since headers were opened".into());
            }
        }
        Ok(())
    }

    fn manifest_path(&self) -> PathBuf {
        self.cache.root.join(format!("{}.json", self.slot))
    }

    pub(super) fn hasher(&self) -> blake3::Hasher {
        let mut hash = blake3::Hasher::new();
        hash.update(b"qwen-v41-converted-source-ranges-v1\0");
        hash.update(self.spec.as_bytes());
        hash
    }

    pub(super) fn lookup(&self, ctx: &MetalContext) -> Result<Option<MetalTensor>, String> {
        let Some(file) = self.open_buffer()? else {
            return Ok(None);
        };
        // Only this private cache's publisher writes these files, before manifest publication.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
        self.validate_sources()?;
        backing::mapped_tensor(ctx, mmap, self.shape.clone(), self.dtype, self.bytes).map(Some)
    }

    pub(super) fn open_buffer(&self) -> Result<Option<File>, String> {
        self.validate_sources()?;
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(self.manifest_path())
        {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.to_string()),
        };
        if file.metadata().map_err(|e| e.to_string())?.len() > 1 << 20 {
            return Err("oversized converted cache manifest".into());
        }
        let manifest: serde_json::Value =
            serde_json::from_reader(file).map_err(|e| e.to_string())?;
        let body = manifest["body"]
            .as_str()
            .ok_or("missing cache manifest body")?;
        if manifest["checksum"].as_str() != Some(blake3::hash(body.as_bytes()).to_hex().as_str()) {
            return Err("converted cache manifest checksum mismatch".into());
        }
        let body: serde_json::Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
        if body["slot"] != self.slot || body["spec"] != self.spec {
            return Err("converted cache version/shape/source mismatch".into());
        }
        let name = body["buffer"].as_str().ok_or("missing cached buffer")?;
        let content = body["content"]
            .as_str()
            .ok_or("missing source content identity")?;
        if checkpoint_fs::parse_hex_32(content).is_none()
            || !name.starts_with(content)
            || Path::new(name).components().count() != 1
            || !matches!(
                Path::new(name).components().next(),
                Some(Component::Normal(_))
            )
        {
            return Err("invalid converted buffer identity/path".into());
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(self.cache.root.join(name))
            .map_err(|e| e.to_string())?;
        let stamp = source_stamp(&file).map_err(|e| e.to_string())?;
        if stamp.size != backing::padded_bytes(self.bytes)? || body["stamp"] != format!("{stamp:?}")
        {
            return Err("converted buffer changed or truncated".into());
        }
        self.validate_sources()?;
        Ok(Some(file))
    }

    pub(super) fn publish(
        &self,
        ctx: &MetalContext,
        writer: backing::BackingWriter,
        hash: blake3::Hasher,
    ) -> Result<MetalTensor, String> {
        self.validate_sources()?;
        let digest = hash.finalize();
        let content = digest.to_hex().to_string();
        let unique = checkpoint_fs::unique_temp_path(&self.cache.root, digest.as_bytes());
        let name = format!(
            "{content}-{}.buffer",
            unique.file_name().unwrap().to_string_lossy()
        );
        let buffer = self.cache.root.join(&name);
        writer.publish(&buffer)?;
        let stamp = source_stamp(&File::open(&buffer).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let body = serde_json::json!({"slot":self.slot, "spec":self.spec, "content":content,
            "buffer":name, "stamp":format!("{stamp:?}")})
        .to_string();
        let manifest = serde_json::json!({"checksum":blake3::hash(body.as_bytes()).to_hex().to_string(), "body":body});
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&unique)
            .map_err(|e| e.to_string())?;
        let result = (|| {
            file.write_all(manifest.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            self.validate_sources()?;
            checkpoint_fs::sync_directory(&self.cache.root).map_err(|e| e.to_string())?;
            std::fs::rename(&unique, self.manifest_path()).map_err(|e| e.to_string())?;
            checkpoint_fs::sync_directory(&self.cache.root).map_err(|e| e.to_string())?;
            self.lookup(ctx)?
                .ok_or("published cache entry disappeared".into())
        })();
        let _ = std::fs::remove_file(unique);
        result
    }
}
