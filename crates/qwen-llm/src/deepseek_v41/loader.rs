//! Runtime loader for the old REAP-272E checkpoint, not the strict 1017 profile.
//! Source headers are not proof of repository/revision authenticity. Keep source
//! files immutable while a store is alive. No whole-file mappings are created.

use super::source::{MAX_HEADER_BYTES, SourceDtype, SourceHeader, TensorDescriptor};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    ops::Range,
    os::unix::fs::FileExt,
    path::{Component, Path},
    sync::Arc,
};

#[path = "loader/backing.rs"]
mod backing;
#[path = "loader/resident.rs"]
mod resident;
pub use resident::{ByteCounters, LoadLimits, PackedBank, ResidentWeights};

pub const OLD_REVISION: &str = "f792019bcb369768220f01088f0f7af230786982";
pub const LAYERS: usize = 40;
pub const EXPERTS: usize = 272;
pub const HIDDEN: usize = 5120;
pub const INTERMEDIATE: usize = 2304;
pub const MAX_READ_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
pub struct StoredTensor {
    pub source_file: String,
    pub descriptor: TensorDescriptor,
    file: Arc<File>,
}

/// An index of completed local shards. Missing shards are reported, not opened;
/// this permits header inspection during download. Malformed present shards fail.
pub struct SourceStore {
    tensors: BTreeMap<String, StoredTensor>,
    pub missing_shards: Vec<String>,
    pub missing_tensors: BTreeMap<String, String>,
}

fn canonical(name: &str) -> String {
    name.strip_prefix("model.")
        .unwrap_or(name)
        .replace("self_attn", "attn")
        .replace("mlp", "ffn")
        .replace("weight_scale_inv", "scale")
        .replace("e_score_correction_bias", "bias")
}

fn safe_path(root: &Path, name: &str) -> Result<std::path::PathBuf, String> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(format!("unsafe shard path: {name}"));
    }
    let path = root.join(path);
    if path.exists()
        && !path
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(root)
    {
        return Err(format!("shard escapes root: {name}"));
    }
    Ok(path)
}

impl SourceStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_impl(root.as_ref(), None)
    }

    /// Open the original-base Engram pair even when installed alongside a REAP
    /// index that does not list those shards. Also loads their small projections.
    pub fn open_engram(root: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_impl(
            root.as_ref(),
            Some(&[
                "model-00047-of-00048.safetensors",
                "model-00048-of-00048.safetensors",
            ]),
        )
    }

    fn open_impl(root: &Path, selected: Option<&[&str]>) -> Result<Self, String> {
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let index_path = safe_path(&root, "model.safetensors.index.json")?;
        let mut indexed = BTreeMap::<String, String>::new();
        let mut shards = std::collections::BTreeSet::new();
        if let Some(selected) = selected {
            for &name in selected {
                shards.insert(name.to_owned());
            }
        } else if index_path.exists() {
            let file = File::open(index_path).map_err(|e| e.to_string())?;
            if file.metadata().map_err(|e| e.to_string())?.len() > MAX_READ_BYTES as u64 {
                return Err("oversized safetensors index".into());
            }
            let mut text = String::new();
            file.take(MAX_READ_BYTES as u64 + 1)
                .read_to_string(&mut text)
                .map_err(|e| e.to_string())?;
            if text.len() > MAX_READ_BYTES {
                return Err("oversized safetensors index".into());
            }
            let value = crate::tool_schema::decode_json(&text)?;
            for (key, value) in value["weight_map"]
                .as_object()
                .ok_or("missing weight_map")?
            {
                let shard = value.as_str().ok_or("non-string shard")?;
                safe_path(&root, shard)?;
                if !shard.ends_with(".safetensors") {
                    return Err("invalid shard extension".into());
                }
                if indexed.insert(canonical(key), shard.into()).is_some() {
                    return Err("canonical tensor collision".into());
                }
                shards.insert(shard.to_owned());
            }
        } else {
            for entry in std::fs::read_dir(&root).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "non-UTF8 filename")?;
                if name.ends_with(".safetensors") {
                    shards.insert(name);
                }
            }
        }
        let mut store = Self {
            tensors: BTreeMap::new(),
            missing_shards: Vec::new(),
            missing_tensors: BTreeMap::new(),
        };
        for shard in shards {
            let path = safe_path(&root, &shard)?;
            let file = match File::open(path) {
                Ok(file) => Arc::new(file),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    store.missing_shards.push(shard);
                    continue;
                }
                Err(e) => return Err(e.to_string()),
            };
            let metadata = file.metadata().map_err(|e| e.to_string())?;
            if !metadata.is_file() {
                return Err("shard is not a regular file".into());
            }
            let mut prefix = [0; 8];
            file.read_exact_at(&mut prefix, 0)
                .map_err(|e| format!("{shard}: {e}"))?;
            let len = u64::from_le_bytes(prefix);
            if len == 0 || len > MAX_HEADER_BYTES || len % 8 != 0 {
                return Err(format!("{shard}: invalid header length"));
            }
            let mut bytes = vec![0; len as usize + 8];
            bytes[..8].copy_from_slice(&prefix);
            file.read_exact_at(&mut bytes[8..], 8)
                .map_err(|e| e.to_string())?;
            let header = SourceHeader::parse(&shard, &bytes, metadata.len())?;
            for (key, descriptor) in header.descriptors() {
                let key = canonical(key);
                if !indexed.is_empty() && indexed.get(&key) != Some(&shard) {
                    return Err(format!("tensor/index mismatch: {key}"));
                }
                if store
                    .tensors
                    .insert(
                        key.clone(),
                        StoredTensor {
                            source_file: shard.clone(),
                            descriptor: descriptor.clone(),
                            file: file.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(format!("duplicate tensor: {key}"));
                }
            }
        }
        for (key, shard) in indexed {
            if store.missing_shards.contains(&shard) {
                store.missing_tensors.insert(key, shard);
                continue;
            }
            if !store.missing_shards.contains(&shard) && !store.tensors.contains_key(&key) {
                return Err(format!("indexed tensor absent: {key}"));
            }
        }
        Ok(store)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = (&str, &StoredTensor)> {
        self.tensors
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    pub fn tensor(&self, name: &str) -> Result<&StoredTensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor: {name}"))
    }

    fn stream_without_cache(&self) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            let mut seen = std::collections::BTreeSet::new();
            for tensor in self.tensors.values() {
                let fd = tensor.file.as_raw_fd();
                if seen.insert(fd) {
                    // SAFETY: the descriptor stays open; only its read-cache policy changes.
                    if unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) } == -1 {
                        return Err(format!(
                            "uncached reads for {}: {}",
                            tensor.source_file,
                            std::io::Error::last_os_error()
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Read a tensor-relative byte range; never allocate or read more than 32 MiB.
    pub fn bounded_read(&self, name: &str, range: Range<u64>) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(name)?;
        let len = range
            .end
            .checked_sub(range.start)
            .ok_or("reversed byte range")?;
        if len > MAX_READ_BYTES as u64
            || range.end > tensor.descriptor.file_bytes.end - tensor.descriptor.file_bytes.start
        {
            return Err("read exceeds tensor or staging bound".into());
        }
        let mut bytes = vec![0; len as usize];
        tensor
            .file
            .read_exact_at(&mut bytes, tensor.descriptor.file_bytes.start + range.start)
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(bytes)
    }

    /// Row-major F32 output, bounded to 32 MiB. FP8 projections use 32x32
    /// scales; Engram uses one scale per row and 32 columns. No FP4 expansion.
    pub fn decode_rows(
        &self,
        name: &str,
        rows: Range<u64>,
        engram: bool,
    ) -> Result<Vec<f32>, String> {
        let d = &self.tensor(name)?.descriptor;
        let (height, width) = row_geometry(d)?;
        let count = rows.end.checked_sub(rows.start).ok_or("reversed rows")?;
        if rows.end > height
            || count
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or("row extent overflow")?
                > MAX_READ_BYTES as u64
        {
            return Err("rows exceed tensor or staging bound".into());
        }
        let scale_name = name.strip_suffix("weight").map(|p| format!("{p}scale"));
        let scale = if d.dtype == SourceDtype::F8E4M3 {
            let key = scale_name.ok_or("FP8 tensor has no weight suffix")?;
            let sd = &self.tensor(&key)?.descriptor;
            let sh = if engram { height } else { height.div_ceil(32) };
            let sw = width.div_ceil(32);
            if d.shape.len() != 2 || sd.dtype != SourceDtype::F8E8M0 || sd.shape != [sh, sw] {
                return Err(format!("invalid FP8 scale geometry: {key}"));
            }
            let start = if engram { rows.start } else { rows.start / 32 };
            let end = if engram {
                rows.end
            } else {
                rows.end.div_ceil(32)
            };
            Some((self.bounded_read(&key, start * sw..end * sw)?, start, sw))
        } else {
            None
        };
        if !matches!(
            d.dtype,
            SourceDtype::Bf16 | SourceDtype::F32 | SourceDtype::F8E4M3
        ) {
            return Err("unsupported dense dtype (packed experts must stay packed)".into());
        }
        let stride = width
            .checked_mul(d.dtype.byte_width())
            .ok_or("row stride overflow")?;
        let raw = self.bounded_read(name, rows.start * stride..rows.end * stride)?;
        let mut output = Vec::with_capacity((count * width) as usize);
        for (i, bytes) in raw.chunks_exact(d.dtype.byte_width() as usize).enumerate() {
            let value = match d.dtype {
                SourceDtype::Bf16 => {
                    f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16)
                }
                SourceDtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
                SourceDtype::F8E4M3 => {
                    let (scales, start, sw) = scale.as_ref().unwrap();
                    let row = rows.start + i as u64 / width;
                    let sr = if engram { row } else { row / 32 } - start;
                    fp8(bytes[0]) * ue8m0(scales[(sr * sw + (i as u64 % width) / 32) as usize])
                }
                _ => unreachable!(),
            };
            if !value.is_finite() {
                return Err(format!("non-finite decoded weight: {name}"));
            }
            output.push(value);
        }
        Ok(output)
    }
}

fn row_geometry(d: &TensorDescriptor) -> Result<(u64, u64), String> {
    let height = *d.shape.first().ok_or("scalar tensor unsupported")?;
    let width = d.shape[1..]
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b).ok_or("row width overflow"))?;
    if height == 0 || width == 0 {
        return Err("empty tensor unsupported".into());
    }
    Ok((height, width))
}

fn ue8m0(code: u8) -> f32 {
    if code == 255 {
        f32::NAN
    } else if code == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits((code as u32) << 23)
    }
}

fn fp8(code: u8) -> f32 {
    let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 15 && mantissa == 7 {
        return f32::NAN;
    }
    sign * if exponent == 0 {
        mantissa as f32 / 512.0
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
    }
}

/// Holds original-base file handles, not an expanded table or mmap. Hashing and
/// tokenizer compression belong to forward; callers supply already-hashed IDs.
pub struct DiskEngram {
    source: Arc<SourceStore>,
}

impl DiskEngram {
    pub fn new(source: Arc<SourceStore>) -> Result<Self, String> {
        for (layer, rows) in [(1, 384006168), (14, 384016682)] {
            for (suffix, dtype, width) in [
                ("weight", SourceDtype::F8E4M3, 256),
                ("scale", SourceDtype::F8E8M0, 8),
            ] {
                let key = format!("layers.{layer}.engram.embed.{suffix}");
                let d = &source.tensor(&key)?.descriptor;
                if d.dtype != dtype || d.shape != [rows, width] {
                    return Err(format!("invalid old Engram tensor: {key}"));
                }
            }
        }
        Ok(Self { source })
    }

    pub fn decode_rows(&self, layer: usize, ids: &[u64]) -> Result<Vec<f32>, String> {
        if ![1, 14].contains(&layer) || ids.len() > MAX_READ_BYTES / (256 * 4) {
            return Err("invalid Engram layer or row request size".into());
        }
        let name = format!("layers.{layer}.engram.embed.weight");
        let height = self.source.tensor(&name)?.descriptor.shape[0];
        if ids.iter().any(|&id| id >= height) {
            return Err("Engram row outside table".into());
        }
        let mut result = Vec::with_capacity(ids.len() * 256);
        for &id in ids {
            result.extend(self.source.decode_rows(&name, id..id + 1, true)?);
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "loader/tests.rs"]
mod tests;
