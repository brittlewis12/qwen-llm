//! GGUF v3 mmap-only loader.
//!
//! Parsing is delegated to `gguf-rs` (Britt's own crate, in `~/code/gguf`)
//! for metadata + tensor-table semantics. We layer:
//!
//! 1. Independent `Mmap`s of the GGUF file(s) so kernels read tensor bytes
//!    directly out of the page cache. **No tensor data is ever copied at
//!    load time.** llama.cpp-style split GGUFs are kept as multiple shard
//!    mmaps behind one logical tensor table.
//! 2. A second pass over the on-disk header to recover the tensor-data
//!    start offset (gguf-rs's parser is streaming and doesn't expose it).
//! 3. Defense-in-depth validation: magic + version checks, bounds checks
//!    on every metadata/table read, and post-load validation that every
//!    declared tensor lies fully inside the file. We assume the GGUF file
//!    is potentially adversarial — a malicious file must produce
//!    [`GgufError`], never a panic, OOB read, or division-by-zero.
//!
//! The bundled `gguf-rs` `MmapGGUF` would have served, but its
//! implementation does `mmap[4..].to_vec()` (a full-file copy into a
//! `Cursor`) — fine for its 5 MB test fixture, catastrophic for a 27 GB
//! GGUF. We avoid that path entirely.

use crate::checkpoint_identity::{SourceStamp, source_stamp};
use crate::tensor::{GgmlType, TensorDesc, ggml_type_layout_raw};
use gguf_rs::{GGUFContainer, GGUFModel};
use memmap2::Mmap;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// "GGUF" in little-endian (only LE is supported; see [`open`] preconditions).
const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const SPLIT_NO_KEY: &str = "split.no";
const SPLIT_COUNT_KEY: &str = "split.count";
const SPLIT_TENSORS_COUNT_KEY: &str = "split.tensors.count";
/// Real split counts are tiny; this bound keeps adversarial metadata from
/// driving enormous path vectors or file-open loops before we can fail safely.
const MAX_SPLITS: u64 = 1024;
/// Hard cap on `general.alignment`. The on-disk default is 32; values up to
/// the page size are reasonable. Anything larger is almost certainly a
/// malicious / corrupted file.
const MAX_ALIGNMENT: u64 = 65536;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("decode: {0}")]
    Decode(String),
    #[error("missing required metadata key: {0}")]
    MissingKey(&'static str),
}

impl From<anyhow::Error> for GgufError {
    fn from(e: anyhow::Error) -> Self {
        GgufError::Decode(e.to_string())
    }
}

/// One opened GGUF shard.
///
/// Split GGUFs are a set of complete GGUF containers, each with its own
/// header, tensor table, tensor-data start, and mmap. Tensor descriptors carry
/// a shard index so the public [`GgufFile::slice`] API can stay one logical
/// model view.
#[allow(dead_code)] // Debug is used by tests via expect_err
pub struct GgufShard {
    pub path: PathBuf,
    /// Original file description backing both the mmap and metadata parser.
    /// Retained so later identity checks can use fstat without re-resolving
    /// the display path.
    pub(crate) file: Arc<File>,
    pub(crate) mmap: Arc<Mmap>,
    pub(crate) source_stamp: SourceStamp,
    /// Absolute byte offset of the start of the tensor-data section.
    /// `TensorDesc.data_offset` values for this shard include this.
    pub tensor_data_start: u64,
    /// Tensor-data alignment in bytes (from `general.alignment`, default 32).
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufShardStamp {
    pub shard_idx: usize,
    pub path: PathBuf,
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub ctime_sec: i64,
    pub ctime_nsec: i64,
}

impl GgufShard {
    /// Length of the mapped file in bytes.
    pub fn mmap_len(&self) -> usize {
        self.mmap.len()
    }

    /// Read-only view over the whole mapped file. Used by external
    /// warmup / diagnostic tooling that needs to walk the mapping
    /// without going through the tensor table. Not on the hot path.
    pub fn mmap_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Pass an [`memmap2::Advice`] hint to the kernel for this shard's
    /// mapping. Callers can use this to signal expected access patterns
    /// (e.g. `WillNeed` before a warmup phase) without needing access to
    /// the underlying `Arc<Mmap>`.
    pub fn advise(&self, advice: memmap2::Advice) -> std::io::Result<()> {
        self.mmap.advise(advice)
    }
}

/// One logical GGUF model, backed by one or more mmap'd GGUF shards.
///
/// * `shards` are held for the full lifetime; tensor slices reference them.
/// * `tensors` is the unified tensor namespace across all shards.
/// * `model` is the gguf-rs decoded view of shard 0 metadata. llama.cpp's
///   `gguf-split` stores full model/tokenizer metadata only in shard 0; later
///   shard metadata is used only during split validation.
#[allow(dead_code)]
pub struct GgufFile {
    pub shards: Vec<GgufShard>,
    pub tensors: Vec<TensorDesc>,
    pub model: GGUFModel,
}

struct LoadedShard {
    shard: GgufShard,
    tensors: Vec<TensorDesc>,
    model: GGUFModel,
}

impl GgufFile {
    /// Open and validate a GGUF v3 file.
    ///
    /// Returns [`GgufError`] (never panics) on:
    /// * I/O error / file-too-small
    /// * Bad magic (not "GGUF")
    /// * Unsupported version (v1, v2, BE host, anything other than v3 LE)
    /// * Malformed metadata or tensor table (truncated, type tag out of range)
    /// * Out-of-bounds tensor declaration (offset + size > file size, or
    ///   offset that wraps `u64`)
    /// * Pathological alignment (0, or > [`MAX_ALIGNMENT`])
    ///
    /// Treat the input file as untrusted.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();

        let first = open_one_shard(path, 0)?;
        let split_count = metadata_u64(&first.model, SPLIT_COUNT_KEY).unwrap_or(0);
        if split_count <= 1 {
            return Ok(Self {
                shards: vec![first.shard],
                tensors: first.tensors,
                model: first.model,
            });
        }
        validate_split_count(split_count, path)?;

        let split_no = metadata_u64(&first.model, SPLIT_NO_KEY).ok_or_else(|| {
            GgufError::Decode(format!(
                "missing {SPLIT_NO_KEY:?} in split GGUF {}",
                path.display()
            ))
        })?;
        if split_no != 0 {
            return Err(GgufError::Decode(format!(
                "illegal split file idx {split_no} (file: {}), model must be loaded with the first split",
                path.display()
            )));
        }
        let split_tensors_count =
            metadata_u64(&first.model, SPLIT_TENSORS_COUNT_KEY).ok_or_else(|| {
                GgufError::Decode(format!(
                    "missing {SPLIT_TENSORS_COUNT_KEY:?} in split GGUF {}",
                    path.display()
                ))
            })?;
        validate_split_tensors_count(split_tensors_count, path)?;

        let split_paths = infer_split_paths(path, split_count)?;
        let split_count_usize = usize::try_from(split_count).map_err(|_| {
            GgufError::Decode(format!("split.count {split_count} does not fit in usize"))
        })?;
        let split_tensors_capacity = usize::try_from(split_tensors_count).map_err(|_| {
            GgufError::Decode(format!(
                "split.tensors.count {split_tensors_count} does not fit in usize"
            ))
        })?;

        let mut shards = Vec::with_capacity(split_count_usize);
        let mut tensors = Vec::with_capacity(split_tensors_capacity);
        ensure_split_extend_within_declared_count(0, first.tensors.len(), split_tensors_count)?;
        shards.push(first.shard);
        tensors.extend(first.tensors);
        let model = first.model;

        for (idx, split_path) in split_paths.iter().enumerate().skip(1) {
            let loaded = open_one_shard(split_path, idx)?;
            validate_split_shard_metadata(
                &loaded.model,
                split_path,
                idx as u64,
                split_count,
                split_tensors_count,
            )?;
            ensure_split_extend_within_declared_count(
                tensors.len(),
                loaded.tensors.len(),
                split_tensors_count,
            )?;
            shards.push(loaded.shard);
            tensors.extend(loaded.tensors);
        }

        validate_unified_tensor_table(&tensors, split_tensors_count)?;
        Ok(Self {
            shards,
            tensors,
            model,
        })
    }

    /// Parse an already-opened single-shard GGUF.
    ///
    /// `diagnostic_path` is never resolved; it is retained only for errors and
    /// diagnostics. Both the mmap and metadata parser are bound to `file`.
    pub fn from_opened_file(
        file: File,
        diagnostic_path: impl Into<PathBuf>,
    ) -> Result<Self, GgufError> {
        let diagnostic_path = diagnostic_path.into();
        let loaded = open_one_shard_file(file, &diagnostic_path, 0)?;
        let split_count = metadata_u64(&loaded.model, SPLIT_COUNT_KEY).unwrap_or(0);
        if split_count > 1 {
            return Err(GgufError::Decode(format!(
                "opened-file constructor requires one shard, but split.count is {split_count} in {}",
                diagnostic_path.display()
            )));
        }
        Ok(Self {
            shards: vec![loaded.shard],
            tensors: loaded.tensors,
            model: loaded.model,
        })
    }

    pub fn primary_shard(&self) -> &GgufShard {
        &self.shards[0]
    }

    pub fn total_mapped_len(&self) -> usize {
        self.shards.iter().map(|shard| shard.mmap.len()).sum()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_mapped_lengths(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.mmap.len()).collect()
    }

    pub(crate) fn retained_shard_mmap(&self, shard_idx: usize) -> Option<Arc<Mmap>> {
        self.shards
            .get(shard_idx)
            .map(|shard| Arc::clone(&shard.mmap))
    }

    /// Slice into the mmap for `desc`. Slice lifetime is tied to `&self`.
    ///
    /// Public callers get a fallible boundary so forged or stale descriptors
    /// cannot panic the process.
    pub fn try_slice(&self, desc: &TensorDesc) -> Result<&[u8], GgufError> {
        let Some(shard) = self.shards.get(desc.shard_idx) else {
            return Err(GgufError::Decode(format!(
                "tensor {:?} references missing shard {}",
                desc.name, desc.shard_idx
            )));
        };
        let start = usize::try_from(desc.data_offset).map_err(|_| {
            GgufError::Decode(format!(
                "tensor {:?} data_offset {} does not fit usize",
                desc.name, desc.data_offset
            ))
        })?;
        let len = usize::try_from(desc.n_bytes).map_err(|_| {
            GgufError::Decode(format!(
                "tensor {:?} n_bytes {} does not fit usize",
                desc.name, desc.n_bytes
            ))
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            GgufError::Decode(format!("tensor {:?} slice endpoint overflow", desc.name))
        })?;
        if end > shard.mmap.len() {
            return Err(GgufError::Decode(format!(
                "tensor {:?} range [{}..{}) exceeds shard {} length {}",
                desc.name,
                start,
                end,
                desc.shard_idx,
                shard.mmap.len()
            )));
        }
        Ok(&shard.mmap[start..end])
    }

    /// Return a validated byte range from one mapped shard.
    pub fn try_shard_range(
        &self,
        shard_idx: usize,
        offset: u64,
        length: usize,
    ) -> Result<&[u8], GgufError> {
        let Some(shard) = self.shards.get(shard_idx) else {
            return Err(GgufError::Decode(format!(
                "range references missing shard {shard_idx}"
            )));
        };
        let start = usize::try_from(offset)
            .map_err(|_| GgufError::Decode(format!("range offset {offset} does not fit usize")))?;
        let end = start.checked_add(length).ok_or_else(|| {
            GgufError::Decode(format!(
                "range endpoint overflows: offset={offset} length={length}"
            ))
        })?;
        if end > shard.mmap.len() {
            return Err(GgufError::Decode(format!(
                "range [{start}..{end}) exceeds shard {shard_idx} length {}",
                shard.mmap.len()
            )));
        }
        Ok(&shard.mmap[start..end])
    }

    /// Read an exact validated range from the descriptor retained for a shard.
    ///
    /// This deliberately does not reopen `GgufShard::path`: callers keep the
    /// same vnode identity that was parsed, validated, and memory-mapped.
    pub fn read_shard_exact_at(
        &self,
        shard_idx: usize,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<(), GgufError> {
        let Some(shard) = self.shards.get(shard_idx) else {
            return Err(GgufError::Decode(format!(
                "range references missing shard {shard_idx}"
            )));
        };
        let length = u64::try_from(destination.len()).map_err(|_| {
            GgufError::Decode(format!(
                "range length {} does not fit u64",
                destination.len()
            ))
        })?;
        let end = offset.checked_add(length).ok_or_else(|| {
            GgufError::Decode(format!(
                "range endpoint overflows: offset={offset} length={length}"
            ))
        })?;
        if end > shard.mmap.len() as u64 {
            return Err(GgufError::Decode(format!(
                "range [{offset}..{end}) exceeds shard {shard_idx} length {}",
                shard.mmap.len()
            )));
        }

        let mut read = 0usize;
        while read < destination.len() {
            let read_offset = offset
                .checked_add(read as u64)
                .ok_or_else(|| GgufError::Decode("range read offset overflow".to_string()))?;
            match shard.file.read_at(&mut destination[read..], read_offset) {
                Ok(0) => {
                    return Err(GgufError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("unexpected EOF in shard {shard_idx} at {read_offset}"),
                    )));
                }
                Ok(bytes) => read += bytes,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(GgufError::Io(error)),
            }
        }
        Ok(())
    }

    /// Revalidate and report the exact retained descriptor for every shard.
    ///
    /// This uses `fstat` through each retained `File`; it never re-resolves the
    /// display path. The complete stamp must still match the stamp captured
    /// while the GGUF was opened, and its size must match the live mmap.
    pub fn revalidate_retained_shard_stamps(&self) -> Result<Vec<GgufShardStamp>, GgufError> {
        self.shards
            .iter()
            .enumerate()
            .map(|(shard_idx, shard)| {
                let current = source_stamp(shard.file.as_ref()).map_err(GgufError::Io)?;
                if current != shard.source_stamp || current.size != shard.mmap.len() as u64 {
                    return Err(GgufError::Decode(format!(
                        "retained GGUF shard {shard_idx} changed after load"
                    )));
                }
                Ok(GgufShardStamp {
                    shard_idx,
                    path: shard.path.clone(),
                    device: current.dev,
                    inode: current.ino,
                    size: current.size,
                    mtime_sec: current.mtime_sec,
                    mtime_nsec: current.mtime_nsec,
                    ctime_sec: current.ctime_sec,
                    ctime_nsec: current.ctime_nsec,
                })
            })
            .collect()
    }

    pub(crate) fn slice(&self, desc: &TensorDesc) -> &[u8] {
        self.try_slice(desc)
            .expect("tensor descriptor should have been validated against its GGUF shard")
    }

    /// Find a tensor by name. Common pattern: `find("token_embd.weight")`.
    pub fn find(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Architecture string from `general.architecture` metadata.
    pub fn architecture(&self) -> Option<String> {
        self.model
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// Convenience: lookup a u64-typed metadata value by key.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        metadata_u64(&self.model, key)
    }

    /// Convenience: lookup a string-typed metadata value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.model.metadata().get(key).and_then(|v| v.as_str())
    }

    /// Convenience: lookup a floating-point metadata value by key.
    pub fn get_f64(&self, key: &str) -> Result<Option<f64>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        value
            .as_f64()
            .map(Some)
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not numeric")))
    }

    /// Convenience: lookup a boolean metadata value by key.
    pub fn get_bool(&self, key: &str) -> Result<Option<bool>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        value
            .as_bool()
            .map(Some)
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not a boolean")))
    }

    /// Return the length of an array metadata value without cloning it.
    pub fn get_array_len(&self, key: &str) -> Result<Option<usize>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        value
            .as_array()
            .map(|values| Some(values.len()))
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not an array")))
    }

    /// Convenience: lookup an array-of-u64 typed metadata value by key.
    /// Returns `Ok(None)` if the key is missing, and `Err` if the key exists
    /// but is not a pure array of u64 values.
    pub fn get_u64_array(&self, key: &str) -> Result<Option<Vec<u64>>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        let arr = value
            .as_array()
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not an array")))?;
        let mut out = Vec::with_capacity(arr.len());
        for (idx, value) in arr.iter().enumerate() {
            let parsed = value.as_u64().ok_or_else(|| {
                GgufError::Decode(format!("metadata key {key:?}[{idx}] is not a u64"))
            })?;
            out.push(parsed);
        }
        Ok(Some(out))
    }

    /// Convenience: lookup an array of signed or unsigned integer metadata.
    /// Unsigned values must fit in `i64`.
    pub fn get_i64_array(&self, key: &str) -> Result<Option<Vec<i64>>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        let arr = value
            .as_array()
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not an array")))?;
        let mut out = Vec::with_capacity(arr.len());
        for (idx, value) in arr.iter().enumerate() {
            let parsed = value.as_i64().or_else(|| {
                value
                    .as_u64()
                    .and_then(|unsigned| i64::try_from(unsigned).ok())
            });
            let parsed = parsed.ok_or_else(|| {
                GgufError::Decode(format!(
                    "metadata key {key:?}[{idx}] is not an i64-compatible integer"
                ))
            })?;
            out.push(parsed);
        }
        Ok(Some(out))
    }

    /// Convenience: lookup an array-of-floating-point metadata value by key.
    /// Returns `Ok(None)` if the key is missing, and `Err` if the key exists
    /// but contains a non-numeric value.
    pub fn get_f64_array(&self, key: &str) -> Result<Option<Vec<f64>>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        let arr = value
            .as_array()
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not an array")))?;
        let mut out = Vec::with_capacity(arr.len());
        for (idx, value) in arr.iter().enumerate() {
            let parsed = value.as_f64().ok_or_else(|| {
                GgufError::Decode(format!("metadata key {key:?}[{idx}] is not numeric"))
            })?;
            out.push(parsed);
        }
        Ok(Some(out))
    }

    /// Convenience: lookup an array-of-bool typed metadata value by key.
    /// Returns `Ok(None)` if the key is missing, and `Err` if the key exists
    /// but is not a pure array of bool values.
    pub fn get_bool_array(&self, key: &str) -> Result<Option<Vec<bool>>, GgufError> {
        let Some(value) = self.model.metadata().get(key) else {
            return Ok(None);
        };
        let arr = value
            .as_array()
            .ok_or_else(|| GgufError::Decode(format!("metadata key {key:?} is not an array")))?;
        let mut out = Vec::with_capacity(arr.len());
        for (idx, value) in arr.iter().enumerate() {
            let parsed = value.as_bool().ok_or_else(|| {
                GgufError::Decode(format!("metadata key {key:?}[{idx}] is not a bool"))
            })?;
            out.push(parsed);
        }
        Ok(Some(out))
    }

    /// Convenience: lookup an f32-typed metadata value by key.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.model
            .metadata()
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
    }

    /// Producer-declared end-of-generation token ids.
    ///
    /// Strictly reads what the GGUF KV section declares — no heuristic
    /// name-matching, no fallback to "what we think Qwen probably means".
    /// The producer's choice is authoritative.
    ///
    /// Reads:
    /// * `tokenizer.ggml.eos_token_id`   — scalar OR array (both are
    ///                                     legal per the GGUF spec; some
    ///                                     Llama-3 / Phi GGUFs ship arrays)
    /// * `tokenizer.ggml.eot_token_id`   — optional, scalar. Added when
    ///                                     the producer wants to split
    ///                                     "end of turn" from "end of
    ///                                     pretraining"
    ///
    /// Returns a deduplicated `Vec<i32>` in declaration order
    /// (EOS-array items first, EOT appended if not already present).
    ///
    /// Errors with [`GgufError::MissingKey`] if **neither** key is
    /// declared. A GGUF with no terminator is a sign of a corrupt or
    /// partially-converted file; we surface that loudly rather than
    /// silently picking a default.
    ///
    /// Implementation note: `Vec` (not `SmallVec`) is correct here — this
    /// is read once at init and stored on the decoder; the hot-loop
    /// membership check is over a 1-2 element slice regardless of
    /// backing storage.
    pub fn stop_token_ids(&self) -> Result<Vec<i32>, GgufError> {
        let mut out: Vec<i32> = Vec::with_capacity(2);

        // EOS: scalar OR array. Try scalar first (the common case for
        // Qwen 3.5/3.6), then fall back to array form.
        if let Some(eos) = self.get_u64("tokenizer.ggml.eos_token_id") {
            out.push(token_id_to_i32("tokenizer.ggml.eos_token_id", eos)?);
        } else if let Some(eos_arr) = self.get_u64_array("tokenizer.ggml.eos_token_id")? {
            for id in eos_arr {
                let id = token_id_to_i32("tokenizer.ggml.eos_token_id", id)?;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }

        // EOT: optional, scalar. Append if not already present.
        if let Some(eot) = self.get_u64("tokenizer.ggml.eot_token_id") {
            let eot = token_id_to_i32("tokenizer.ggml.eot_token_id", eot)?;
            if !out.contains(&eot) {
                out.push(eot);
            }
        }

        if out.is_empty() {
            return Err(GgufError::MissingKey("tokenizer.ggml.eos_token_id"));
        }
        Ok(out)
    }
}

fn token_id_to_i32(key: &'static str, value: u64) -> Result<i32, GgufError> {
    i32::try_from(value).map_err(|_| {
        GgufError::Decode(format!(
            "metadata key {key:?} token id {value} does not fit in i32"
        ))
    })
}

fn open_one_shard(path: &Path, shard_idx: usize) -> Result<LoadedShard, GgufError> {
    open_one_shard_file(File::open(path)?, path, shard_idx)
}

fn open_one_shard_file(
    file: File,
    path: &Path,
    shard_idx: usize,
) -> Result<LoadedShard, GgufError> {
    // mmap is the source of truth for tensor data.
    let file = Arc::new(file);
    let baseline_source_stamp = source_stamp(file.as_ref()).map_err(GgufError::Io)?;
    // SAFETY: regular file held for the lifetime of `Self`. Memory mapping a
    // file handed to us by the user is the standard path; if the file is
    // concurrently truncated underneath us we'll SIGBUS on access — that is an
    // OS-level signal we cannot prevent in safe Rust without copying, and
    // would be the user racing themselves.
    let mmap = Arc::new(unsafe { Mmap::map(file.as_ref())? });
    if baseline_source_stamp.size() != mmap.len() as u64 {
        return Err(GgufError::Decode(format!(
            "GGUF size changed while mapping {}",
            path.display()
        )));
    }

    // Validate magic against the mmap directly. The mmap is the *only* path
    // that the rest of this function trusts; the streaming parser is given a
    // separate `File` and its results are reconciled below.
    validate_magic(&mmap)?;
    let version = read_u32_at(&mmap, 4)?;
    if version != 3 {
        return Err(GgufError::Decode(format!(
            "unsupported GGUF version {version} (only v3 supported)"
        )));
    }
    prevalidate_header_before_decode(&mmap)?;

    // Parse through a duplicate of the exact open file description backing the
    // mmap. Re-resolving the path here could bind metadata and tensor bytes to
    // different vnodes if the path were atomically replaced during load.
    let mut parse_file = file.try_clone()?;
    // `File::try_clone` duplicates the descriptor but preserves the shared
    // open-file-description cursor on Darwin. Callers may have consumed a
    // clone while authenticating the file, so the streaming parser must own
    // its starting-position contract rather than inheriting that cursor.
    parse_file.seek(SeekFrom::Start(0))?;
    let mut container = GGUFContainer::new(
        Box::new(BufReader::with_capacity(64 * 1024, parse_file)),
        u64::MAX,
    )?
    .with_input_len(mmap.len() as u64);
    let model = container.decode()?;

    // Read and validate alignment.
    let alignment = read_alignment(&model)?;

    // Independently locate tensor-data start by replaying the header layout
    // against the mmap. This both validates the on-disk structure and yields
    // the offset we need for absolute tensor addressing. Cross-checks the
    // (num_tensors, num_kv) against gguf-rs's parse to catch disagreement.
    let tensor_data_start = locate_tensor_data_start(&mmap, &model, alignment)?;

    // Build TensorDesc list with absolute offsets, validating every tensor
    // lies fully inside this shard with no overflow and no overlap.
    let mmap_len = mmap.len() as u64;
    let mut tensors: Vec<TensorDesc> = Vec::with_capacity(model.tensors().len());
    for t in model.tensors() {
        // gguf-rs stores trailing 1s for unused dims; drop them so
        // `shape.len()` reflects actual rank.
        let mut shape: Vec<u64> = t.shape.to_vec();
        while shape.len() > 1 && *shape.last().unwrap() == 1 {
            shape.pop();
        }

        for &d in &shape {
            if d == 0 {
                return Err(GgufError::Decode(format!(
                    "tensor {:?} has a zero-length dimension: {:?}",
                    t.name, shape
                )));
            }
            if d > MAX_DIMENSION {
                return Err(GgufError::Decode(format!(
                    "tensor {:?} dimension {d} exceeds {MAX_DIMENSION}",
                    t.name
                )));
            }
        }
        let mut elements: u64 = 1;
        for &d in &shape {
            elements = elements.checked_mul(d).ok_or_else(|| {
                GgufError::Decode(format!(
                    "tensor {:?} dimension product overflows u64",
                    t.name
                ))
            })?;
        }
        if elements > MAX_ELEMENTS {
            return Err(GgufError::Decode(format!(
                "tensor {:?} element count {elements} exceeds {MAX_ELEMENTS}",
                t.name
            )));
        }
        validate_tensor_storage_size(&t.name, t.kind, elements, t.size)?;

        // TENSOR-13: t.offset is relative to data start; alignment applies to
        // that. (Equivalent to data_offset % alignment == 0 since
        // tensor_data_start itself is aligned.)
        if t.offset % alignment != 0 {
            return Err(GgufError::Decode(format!(
                "tensor {:?} offset {} not aligned to {}",
                t.name, t.offset, alignment
            )));
        }

        let data_offset = tensor_data_start.checked_add(t.offset).ok_or_else(|| {
            GgufError::Decode(format!(
                "tensor {:?} offset {} overflows when added to data start {}",
                t.name, t.offset, tensor_data_start
            ))
        })?;
        let end = data_offset.checked_add(t.size).ok_or_else(|| {
            GgufError::Decode(format!(
                "tensor {:?} (offset {}, size {}) end overflows u64",
                t.name, data_offset, t.size
            ))
        })?;
        if end > mmap_len {
            return Err(GgufError::Decode(format!(
                "tensor {:?} extends past EOF: end={} file_size={}",
                t.name, end, mmap_len
            )));
        }
        tensors.push(TensorDesc {
            name: t.name.clone(),
            shape,
            dtype: GgmlType::from_raw(t.kind),
            shard_idx,
            data_offset,
            n_bytes: t.size,
        });
    }

    // LAYOUT-03: no two tensor data ranges overlap within this shard. Offsets
    // in different shards are intentionally independent and may be identical.
    let mut sort_idx: Vec<usize> = (0..tensors.len()).collect();
    sort_idx.sort_by_key(|&i| tensors[i].data_offset);
    for w in sort_idx.windows(2) {
        let (a, b) = (&tensors[w[0]], &tensors[w[1]]);
        let a_end = a.data_offset.checked_add(a.n_bytes).ok_or_else(|| {
            GgufError::Decode(format!("tensor {:?} endpoint overflows u64", a.name))
        })?;
        if a_end > b.data_offset {
            return Err(GgufError::Decode(format!(
                "tensors {:?} and {:?} have overlapping data ranges",
                a.name, b.name
            )));
        }
    }

    if source_stamp(file.as_ref()).map_err(GgufError::Io)? != baseline_source_stamp {
        return Err(GgufError::Decode(format!(
            "GGUF changed while loading {}",
            path.display()
        )));
    }

    Ok(LoadedShard {
        shard: GgufShard {
            path: path.to_path_buf(),
            file,
            mmap,
            source_stamp: baseline_source_stamp,
            tensor_data_start,
            alignment,
        },
        tensors,
        model,
    })
}

fn infer_split_paths(path: &Path, split_count: u64) -> Result<Vec<PathBuf>, GgufError> {
    let path_str = path.to_str().ok_or_else(|| {
        GgufError::Decode(format!("split GGUF path is not valid UTF-8: {path:?}"))
    })?;
    let first_suffix = split_suffix(0, split_count);
    let Some(prefix) = path_str.strip_suffix(&first_suffix) else {
        return Err(GgufError::Decode(format!(
            "invalid split file name: {} (expected suffix {first_suffix:?})",
            path.display()
        )));
    };
    if prefix.is_empty() {
        return Err(GgufError::Decode(format!(
            "invalid split file: {}",
            path.display()
        )));
    }
    let split_count_usize = usize::try_from(split_count).map_err(|_| {
        GgufError::Decode(format!("split.count {split_count} does not fit in usize"))
    })?;
    let mut paths = Vec::with_capacity(split_count_usize);
    for idx in 0..split_count {
        paths.push(PathBuf::from(format!(
            "{}{}",
            prefix,
            split_suffix(idx, split_count)
        )));
    }
    Ok(paths)
}

fn split_suffix(split_no: u64, split_count: u64) -> String {
    format!("-{:05}-of-{:05}.gguf", split_no + 1, split_count)
}

fn validate_split_shard_metadata(
    model: &GGUFModel,
    path: &Path,
    expected_idx: u64,
    expected_count: u64,
    expected_total_tensors: u64,
) -> Result<(), GgufError> {
    let got_idx = metadata_u64(model, SPLIT_NO_KEY).ok_or_else(|| {
        GgufError::Decode(format!(
            "missing {SPLIT_NO_KEY:?} in GGUF split {}",
            path.display()
        ))
    })?;
    if got_idx != expected_idx {
        return Err(GgufError::Decode(format!(
            "invalid split file idx: {got_idx} (file: {}), expected {expected_idx}",
            path.display()
        )));
    }

    let got_count = metadata_u64(model, SPLIT_COUNT_KEY).ok_or_else(|| {
        GgufError::Decode(format!(
            "missing {SPLIT_COUNT_KEY:?} in GGUF split {}",
            path.display()
        ))
    })?;
    validate_split_count(got_count, path)?;
    if got_count != expected_count {
        return Err(GgufError::Decode(format!(
            "invalid split count: {got_count} (file: {}), expected {expected_count}",
            path.display()
        )));
    }

    if let Some(got_total) = metadata_u64(model, SPLIT_TENSORS_COUNT_KEY) {
        validate_split_tensors_count(got_total, path)?;
        if got_total != expected_total_tensors {
            return Err(GgufError::Decode(format!(
                "invalid split tensor count: {got_total} (file: {}), expected {expected_total_tensors}",
                path.display()
            )));
        }
    }

    Ok(())
}

fn validate_split_count(split_count: u64, path: &Path) -> Result<(), GgufError> {
    if split_count > MAX_SPLITS {
        return Err(GgufError::Decode(format!(
            "split.count {split_count} in {} exceeds cap {MAX_SPLITS}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_split_tensors_count(total_tensors: u64, path: &Path) -> Result<(), GgufError> {
    if total_tensors > MAX_TENSORS {
        return Err(GgufError::Decode(format!(
            "split.tensors.count {total_tensors} in {} exceeds cap {MAX_TENSORS}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_split_extend_within_declared_count(
    current: usize,
    additional: usize,
    expected_total: u64,
) -> Result<(), GgufError> {
    let next = current.checked_add(additional).ok_or_else(|| {
        GgufError::Decode("split tensor count overflows usize while merging shards".into())
    })?;
    if next as u64 > expected_total {
        return Err(GgufError::Decode(format!(
            "split shards declare {expected_total} tensors but at least {next} were found"
        )));
    }
    Ok(())
}

fn validate_unified_tensor_table(
    tensors: &[TensorDesc],
    expected_total_tensors: u64,
) -> Result<(), GgufError> {
    let mut names = std::collections::HashSet::with_capacity(tensors.len());
    for t in tensors {
        if !names.insert(t.name.as_str()) {
            return Err(GgufError::Decode(format!(
                "invalid model: tensor {:?} is duplicated across GGUF splits",
                t.name
            )));
        }
    }
    if tensors.len() as u64 != expected_total_tensors {
        return Err(GgufError::Decode(format!(
            "corrupted split model: {expected_total_tensors} tensors expected but {} found",
            tensors.len()
        )));
    }
    Ok(())
}

fn validate_tensor_storage_size(
    name: &str,
    kind: u32,
    elements: u64,
    declared_size: u64,
) -> Result<(), GgufError> {
    let (block_size, type_size) = ggml_type_layout_raw(kind).ok_or_else(|| {
        GgufError::Decode(format!(
            "tensor {name:?} declares unsupported GGML type {kind}"
        ))
    })?;
    if block_size == 0 || type_size == 0 {
        return Err(GgufError::Decode(format!(
            "tensor {name:?} declares removed GGML type {kind}"
        )));
    }
    if !elements.is_multiple_of(block_size) {
        return Err(GgufError::Decode(format!(
            "tensor {name:?} element count {elements} is not divisible by GGML block size {block_size} for type {kind}"
        )));
    }
    let expected = elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| {
            GgufError::Decode(format!(
                "tensor {name:?} byte size overflows for type {kind}"
            ))
        })?;
    if declared_size != expected {
        return Err(GgufError::Decode(format!(
            "tensor {name:?} has byte size {declared_size}, expected {expected} for type {kind}"
        )));
    }
    Ok(())
}

/// Replay enough of the GGUF header before handing the file to `gguf-rs` to
/// ensure malformed adversarial files fail as `GgufError`, not as parser
/// panics or unbounded parser work. The full cross-checking still happens in
/// `locate_tensor_data_start` after decode.
fn prevalidate_header_before_decode(mmap: &[u8]) -> Result<(), GgufError> {
    let mut p: usize = 0;

    let magic = read_u32(mmap, &mut p)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    let version = read_u32(mmap, &mut p)?;
    if version != 3 {
        return Err(GgufError::Decode(format!(
            "unsupported GGUF version {version} (only v3 supported)"
        )));
    }

    let num_tensors = read_u64(mmap, &mut p)?;
    let num_kv = read_u64(mmap, &mut p)?;
    if num_tensors > MAX_TENSORS {
        return Err(GgufError::Decode(format!(
            "absurd tensor count: {num_tensors} (cap {MAX_TENSORS})"
        )));
    }
    if num_kv > MAX_KV {
        return Err(GgufError::Decode(format!(
            "absurd kv count: {num_kv} (cap {MAX_KV})"
        )));
    }

    for _ in 0..num_kv {
        let key_len = read_u64(mmap, &mut p)?;
        if key_len == 0 {
            return Err(GgufError::Decode("metadata key has zero length".into()));
        }
        if key_len > MAX_KEY_LEN {
            return Err(GgufError::Decode(format!(
                "metadata key length {key_len} exceeds {MAX_KEY_LEN}"
            )));
        }
        bounds_check(mmap, p, key_len as usize)?;
        std::str::from_utf8(&mmap[p..p + key_len as usize])
            .map_err(|e| GgufError::Decode(format!("metadata key not valid UTF-8: {e}")))?;
        p += key_len as usize;

        let value_type = read_u32(mmap, &mut p)?;
        skip_value(mmap, &mut p, value_type, version)?;
    }

    for _ in 0..num_tensors {
        let name_len = read_u64(mmap, &mut p)?;
        if name_len == 0 {
            return Err(GgufError::Decode("tensor name has zero length".into()));
        }
        if name_len > MAX_TENSOR_NAME_LEN {
            return Err(GgufError::Decode(format!(
                "tensor name length {name_len} exceeds {MAX_TENSOR_NAME_LEN}"
            )));
        }
        bounds_check(mmap, p, name_len as usize)?;
        std::str::from_utf8(&mmap[p..p + name_len as usize])
            .map_err(|e| GgufError::Decode(format!("tensor name not valid UTF-8: {e}")))?;
        p += name_len as usize;

        let n_dims = read_u32(mmap, &mut p)?;
        if n_dims == 0 || n_dims > 4 {
            return Err(GgufError::Decode(format!(
                "tensor declares {n_dims} dimensions (must be 1..=4)"
            )));
        }
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64(mmap, &mut p)?;
            if dim == 0 {
                return Err(GgufError::Decode(
                    "tensor has a zero-length dimension".into(),
                ));
            }
            if dim > MAX_DIMENSION {
                return Err(GgufError::Decode(format!(
                    "tensor dimension {dim} exceeds {MAX_DIMENSION}"
                )));
            }
            elements = elements.checked_mul(dim).ok_or_else(|| {
                GgufError::Decode("tensor dimension product overflows u64".into())
            })?;
        }
        if elements > MAX_ELEMENTS {
            return Err(GgufError::Decode(format!(
                "tensor element count {elements} exceeds {MAX_ELEMENTS}"
            )));
        }
        let kind = read_u32(mmap, &mut p)?;
        if kind >= 40 {
            return Err(GgufError::Decode(format!(
                "tensor declares invalid GGML type {kind}"
            )));
        }
        bounds_check(mmap, p, 8)?; // offset
        p += 8;
    }

    Ok(())
}

/// Read `general.alignment` (default 32). Rejects values that would cause
/// arithmetic anomalies downstream (0, or > [`MAX_ALIGNMENT`]).
fn read_alignment(model: &GGUFModel) -> Result<u64, GgufError> {
    let raw =
        value_as_u64(model.metadata().get("general.alignment")).unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if raw == 0 {
        return Err(GgufError::Decode(
            "general.alignment is 0 (would divide by zero)".into(),
        ));
    }
    if raw > MAX_ALIGNMENT {
        return Err(GgufError::Decode(format!(
            "general.alignment {raw} exceeds {MAX_ALIGNMENT}"
        )));
    }
    Ok(raw)
}

fn value_as_u64(v: Option<&Value>) -> Option<u64> {
    v.and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
    })
}

fn metadata_u64(model: &GGUFModel, key: &str) -> Option<u64> {
    value_as_u64(model.metadata().get(key))
}

/// Validate the leading 4 bytes are the GGUF magic.
fn validate_magic(mmap: &[u8]) -> Result<(), GgufError> {
    if mmap.len() < 4 {
        return Err(GgufError::Decode(format!(
            "file too short: {} bytes",
            mmap.len()
        )));
    }
    let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    Ok(())
}

fn read_u32_at(mmap: &[u8], offset: usize) -> Result<u32, GgufError> {
    if mmap.len() < offset + 4 {
        return Err(GgufError::Decode(format!(
            "file truncated at offset {offset}"
        )));
    }
    Ok(u32::from_le_bytes(
        mmap[offset..offset + 4].try_into().unwrap(),
    ))
}

// POLICY caps. The proper home for these is `gguf-validator` (see
// `~/code/gguf-validator/SPEC.md`); this is a defensive subset of its
// SPEC + POLICY ERROR-severity invariants applied at load time.
//
// Real-world bounds: Qwen3.6-27B has ~640 tensors, 80B-A3B has ~1500;
// tokenizer vocab arrays top out around 250K entries (Qwen3.5/3.6 vocab
// is 248,320). Our caps are looser than the SPEC defaults to avoid
// spurious rejection on unusual but legitimate files, while still
// catching obvious allocation bombs.

/// HDR-03/04: cap on declared tensor + KV counts.
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KV: u64 = 1_000_000;
/// META-02: cap on metadata key length. Real GGUF keys are <= 256 bytes.
const MAX_KEY_LEN: u64 = 1024;
/// TENSOR-01b: GGUF spec name-length limit is 64 bytes (`GGML_MAX_NAME`).
const MAX_TENSOR_NAME_LEN: u64 = 64;
/// META-06: cap on a single string value (chat templates can be large).
const MAX_STRING_VALUE_LEN: u64 = 100 * 1024 * 1024;
/// META-09: cap on metadata array length.
const MAX_ARRAY_LEN: u64 = 10_000_000;
/// TENSOR-07: per-dimension cap.
const MAX_DIMENSION: u64 = 1u64 << 30;
/// TENSOR-09: total element count cap.
const MAX_ELEMENTS: u64 = 1u64 << 40;

/// Locate the byte offset where the tensor-data section starts.
///
/// GGUF on-disk layout (v3, LE):
///
/// ```text
///   u32  magic = "GGUF"
///   u32  version = 3
///   u64  num_tensors
///   u64  num_kv
///   for each kv:
///     string key
///     u32    value_type
///     value (variable)
///   for each tensor info:
///     string name
///     u32    n_dims
///     u64[n_dims] shape
///     u32    type
///     u64    offset (relative to tensor-data start)
///   <padding to alignment>
///   <tensor data>
/// ```
///
/// Strings are: `u64 length` + raw bytes (length is `u32` in v1 only).
///
/// We replay this layout against the mmap to find the offset of "<padding>".
fn locate_tensor_data_start(
    mmap: &[u8],
    model: &GGUFModel,
    alignment: u64,
) -> Result<u64, GgufError> {
    let mut p: usize = 0;

    // u32 magic — re-validate. (This function trusts no upstream parser.)
    let magic = read_u32(mmap, &mut p)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    // u32 version
    let version = read_u32(mmap, &mut p)?;
    if version != 3 {
        return Err(GgufError::Decode(format!(
            "unsupported GGUF version {version} (only v3 supported)"
        )));
    }

    // u64 num_tensors, u64 num_kv
    let num_tensors = read_u64(mmap, &mut p)?;
    let num_kv = read_u64(mmap, &mut p)?;

    if num_tensors > MAX_TENSORS {
        return Err(GgufError::Decode(format!(
            "absurd tensor count: {num_tensors} (cap {MAX_TENSORS})"
        )));
    }
    if num_kv > MAX_KV {
        return Err(GgufError::Decode(format!(
            "absurd kv count: {num_kv} (cap {MAX_KV})"
        )));
    }

    // Sanity-check vs the parsed model.
    if num_tensors != model.num_tensor() {
        return Err(GgufError::Decode(format!(
            "tensor count mismatch: header says {num_tensors}, parser says {}",
            model.num_tensor()
        )));
    }
    if num_kv != model.num_kv() {
        return Err(GgufError::Decode(format!(
            "kv count mismatch: header says {num_kv}, parser says {}",
            model.num_kv()
        )));
    }

    // Skip metadata, validating each KV header (META-01, META-02, META-04,
    // META-05, META-12). Duplicate-key detection materializes the keys.
    let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    for _ in 0..num_kv {
        let key_len = read_u64(mmap, &mut p)?;
        if key_len == 0 {
            return Err(GgufError::Decode("metadata key has zero length".into()));
        }
        if key_len > MAX_KEY_LEN {
            return Err(GgufError::Decode(format!(
                "metadata key length {key_len} exceeds {MAX_KEY_LEN}"
            )));
        }
        bounds_check(mmap, p, key_len as usize)?;
        let key = std::str::from_utf8(&mmap[p..p + key_len as usize])
            .map_err(|e| GgufError::Decode(format!("metadata key not valid UTF-8: {e}")))?
            .to_string();
        if !seen_keys.insert(key.clone()) {
            return Err(GgufError::Decode(format!("duplicate metadata key {key:?}")));
        }
        p += key_len as usize;

        let value_type = read_u32(mmap, &mut p)?;
        skip_value(mmap, &mut p, value_type, version)?;
    }

    // Tensor table (TENSOR-01a/b, TENSOR-03, TENSOR-04, TENSOR-05, TENSOR-13).
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for _ in 0..num_tensors {
        let name_len = read_u64(mmap, &mut p)?;
        if name_len == 0 {
            return Err(GgufError::Decode("tensor name has zero length".into()));
        }
        if name_len > MAX_TENSOR_NAME_LEN {
            return Err(GgufError::Decode(format!(
                "tensor name length {name_len} exceeds {MAX_TENSOR_NAME_LEN}"
            )));
        }
        bounds_check(mmap, p, name_len as usize)?;
        let name = std::str::from_utf8(&mmap[p..p + name_len as usize])
            .map_err(|e| GgufError::Decode(format!("tensor name not valid UTF-8: {e}")))?
            .to_string();
        if !seen_names.insert(name.clone()) {
            return Err(GgufError::Decode(format!("duplicate tensor name {name:?}")));
        }
        p += name_len as usize;

        let n_dims = read_u32(mmap, &mut p)?;
        // GGML hardcodes GGML_MAX_DIMS = 4. Reject anything else.
        if n_dims == 0 || n_dims > 4 {
            return Err(GgufError::Decode(format!(
                "tensor {name:?} declares {n_dims} dimensions (must be 1..=4)"
            )));
        }
        let shape_bytes = 8usize * n_dims as usize;
        bounds_check(mmap, p, shape_bytes)?;
        p += shape_bytes;
        bounds_check(mmap, p, 4)?; // type
        p += 4;
        bounds_check(mmap, p, 8)?; // offset
        p += 8;
    }

    // Pad to alignment. We've validated alignment is in 1..=MAX_ALIGNMENT
    // upstream, so neither div nor mul overflows.
    let p64 = p as u64;
    let aligned = p64
        .checked_add(alignment - 1)
        .ok_or_else(|| GgufError::Decode("header end + alignment overflows u64".into()))?
        / alignment
        * alignment;
    if aligned > mmap.len() as u64 {
        if model.tensors().is_empty() && p == mmap.len() {
            return Ok(p64);
        }
        return Err(GgufError::Decode(format!(
            "aligned header end {aligned} > file size {}",
            mmap.len()
        )));
    }
    Ok(aligned)
}

fn read_u32(mmap: &[u8], p: &mut usize) -> Result<u32, GgufError> {
    bounds_check(mmap, *p, 4)?;
    let v = u32::from_le_bytes(mmap[*p..*p + 4].try_into().unwrap());
    *p += 4;
    Ok(v)
}

fn read_u64(mmap: &[u8], p: &mut usize) -> Result<u64, GgufError> {
    bounds_check(mmap, *p, 8)?;
    let v = u64::from_le_bytes(mmap[*p..*p + 8].try_into().unwrap());
    *p += 8;
    Ok(v)
}

/// Skip a length-prefixed string value (i.e. a metadata STRING value or
/// an element of a STRING-typed array). Caller is responsible for using
/// the right cap when skipping a key (`MAX_KEY_LEN`) or tensor name
/// (`MAX_TENSOR_NAME_LEN`); see those callers in `locate_tensor_data_start`.
fn skip_string(mmap: &[u8], p: &mut usize) -> Result<(), GgufError> {
    let len = read_u64(mmap, p)?;
    if len > MAX_STRING_VALUE_LEN {
        return Err(GgufError::Decode(format!(
            "absurd string length {len} at offset {} (cap {MAX_STRING_VALUE_LEN})",
            *p
        )));
    }
    let len = len as usize;
    bounds_check(mmap, *p, len)?;
    *p += len;
    Ok(())
}

fn skip_value(mmap: &[u8], p: &mut usize, value_type: u32, version: u32) -> Result<(), GgufError> {
    match value_type {
        0 | 1 => {
            // u8 / i8
            bounds_check(mmap, *p, 1)?;
            *p += 1;
        }
        2 | 3 => {
            // u16 / i16
            bounds_check(mmap, *p, 2)?;
            *p += 2;
        }
        4..=6 => {
            // u32 / i32 / f32
            bounds_check(mmap, *p, 4)?;
            *p += 4;
        }
        7 => {
            // bool (one byte on disk per GGUF spec)
            bounds_check(mmap, *p, 1)?;
            *p += 1;
        }
        8 => skip_string(mmap, p)?,
        9 => {
            // array: u32 item_type, length (u64 in v2/v3, u32 in v1), then items
            let item_type = read_u32(mmap, p)?;
            if item_type == 9 {
                return Err(GgufError::Decode(
                    "nested metadata arrays are not supported".into(),
                ));
            }
            let len = if version == 1 {
                read_u32(mmap, p)? as u64
            } else {
                read_u64(mmap, p)?
            };
            if len > MAX_ARRAY_LEN {
                return Err(GgufError::Decode(format!(
                    "absurd array length {len} at offset {} (cap {MAX_ARRAY_LEN})",
                    *p
                )));
            }
            for _ in 0..len {
                skip_value(mmap, p, item_type, version)?;
            }
        }
        10..=12 => {
            // u64 / i64 / f64
            bounds_check(mmap, *p, 8)?;
            *p += 8;
        }
        other => {
            return Err(GgufError::Decode(format!(
                "unknown metadata value type {other}"
            )));
        }
    }
    Ok(())
}

fn bounds_check(mmap: &[u8], p: usize, n: usize) -> Result<(), GgufError> {
    if p.checked_add(n).is_none_or(|end| end > mmap.len()) {
        return Err(GgufError::Decode(format!(
            "out-of-bounds read at offset {p} (len {n}, file size {})",
            mmap.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> std::path::PathBuf {
        // Use the smallest local Qwen3.5 GGUF as a real-world fixture.
        // Falls back to the gguf-rs test fixture if the model isn't present.
        let candidates = [
            "/Users/tito/models/Qwen3.5-0.8B-Q4_0.gguf",
            "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf",
            "/Users/tito/code/gguf/tests/test-le-v3.gguf",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return c.into();
            }
        }
        panic!("no GGUF fixture available");
    }

    #[test]
    fn parses_real_gguf() {
        let path = fixture();
        let g = GgufFile::open(&path).expect("open");
        assert!(!g.tensors.is_empty(), "tensor table is empty");
        let primary = g.primary_shard();
        assert!(primary.tensor_data_start > 0);
        // tensor data start must be aligned.
        assert_eq!(primary.tensor_data_start % primary.alignment, 0);

        // First tensor must begin exactly at tensor_data_start
        // (the smallest relative offset is always 0).
        let min_offset = g
            .tensors
            .iter()
            .filter(|t| t.shard_idx == 0)
            .map(|t| t.data_offset)
            .min()
            .unwrap();
        assert_eq!(min_offset, primary.tensor_data_start);

        // Every tensor must end inside its own shard.
        for t in &g.tensors {
            let end = (t.data_offset + t.n_bytes) as usize;
            let shard_len = g.shards[t.shard_idx].mmap.len();
            assert!(
                end <= shard_len,
                "tensor {} end {} > shard {} file size {}",
                t.name,
                end,
                t.shard_idx,
                shard_len
            );
        }

        eprintln!(
            "[gguf-test] {}: arch={:?}, {} tensors, {} shard(s), primary data starts at {}, mmap total is {} MiB",
            path.display(),
            g.architecture(),
            g.tensors.len(),
            g.shard_count(),
            primary.tensor_data_start,
            g.total_mapped_len() / (1024 * 1024),
        );
    }

    /// Build a minimal valid GGUF v3 byte stream with no metadata and one
    /// tiny F32 tensor of shape [1]. Useful as a base for adversarial
    /// tweaks.
    fn build_minimal_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        b.extend_from_slice(&0u64.to_le_bytes()); // 0 KV
        // tensor info
        let name = b"t";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&1u64.to_le_bytes()); // shape[0] = 1
        b.extend_from_slice(&0u32.to_le_bytes()); // type F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset 0
        // align to 32, then 4 bytes of f32 payload
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b
    }

    fn build_unpadded_nonempty_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        push_string(&mut bytes, "t");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes
    }

    enum TestKv<'a> {
        U16(&'a str, u16),
        U64(&'a str, u64),
        I32(&'a str, i32),
    }

    struct TestTensor<'a> {
        name: &'a str,
        value: f32,
    }

    fn push_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn build_test_gguf(kvs: &[TestKv<'_>], tensors: &[TestTensor<'_>]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            while data.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
                data.push(0);
            }
            offsets.push(data.len() as u64);
            data.extend_from_slice(&t.value.to_le_bytes());
        }

        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

        for kv in kvs {
            match kv {
                TestKv::U16(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&2u32.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::U64(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::I32(key, value) => {
                    push_string(&mut b, key);
                    b.extend_from_slice(&5u32.to_le_bytes());
                    b.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        for (i, t) in tensors.iter().enumerate() {
            push_string(&mut b, t.name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&offsets[i].to_le_bytes());
        }

        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&data);
        b
    }

    fn build_split_shard(
        split_no: u16,
        split_count: u16,
        total_tensors: i32,
        tensors: &[TestTensor<'_>],
    ) -> Vec<u8> {
        let kvs = [
            TestKv::U16(SPLIT_NO_KEY, split_no),
            TestKv::U16(SPLIT_COUNT_KEY, split_count),
            TestKv::I32(SPLIT_TENSORS_COUNT_KEY, total_tensors),
        ];
        build_test_gguf(&kvs, tensors)
    }

    fn build_unpadded_empty_split_shard(
        split_no: u16,
        split_count: u16,
        total_tensors: i32,
    ) -> Vec<u8> {
        let kvs = [
            TestKv::U16(SPLIT_NO_KEY, split_no),
            TestKv::U16(SPLIT_COUNT_KEY, split_count),
            TestKv::I32(SPLIT_TENSORS_COUNT_KEY, total_tensors),
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for kv in &kvs {
            match kv {
                TestKv::U16(key, value) => {
                    push_string(&mut bytes, key);
                    bytes.extend_from_slice(&2u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::I32(key, value) => {
                    push_string(&mut bytes, key);
                    bytes.extend_from_slice(&5u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                TestKv::U64(_, _) => unreachable!("split metadata has no u64 values"),
            }
        }
        bytes
    }

    fn build_overlapping_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        for name in ["a", "b"] {
            push_string(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&1u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b
    }

    fn build_bad_rank_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        push_string(&mut b, "bad");
        b.extend_from_slice(&5u32.to_le_bytes());
        for _ in 0..5 {
            b.extend_from_slice(&1u64.to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b
    }

    fn build_bad_dimension_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        push_string(&mut b, "bad");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&(MAX_DIMENSION + 1).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b
    }

    fn build_bad_type_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        push_string(&mut b, "bad");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b
    }

    fn build_bad_quant_alignment_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        push_string(&mut b, "bad_q");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&12u32.to_le_bytes()); // Q4_K requires 256-element blocks
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            b.push(0);
        }
        b.extend_from_slice(&[0u8; 4]);
        b
    }

    fn build_nested_array_metadata_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        push_string(&mut b, "nested");
        b.extend_from_slice(&9u32.to_le_bytes()); // array
        b.extend_from_slice(&9u32.to_le_bytes()); // item_type = array
        b.extend_from_slice(&1u64.to_le_bytes()); // one nested item
        b
    }

    fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
        use std::io::Write;
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "qwen-gguf-test-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(bytes).unwrap();
        tmp
    }

    fn write_file(path: &std::path::Path, bytes: &[u8]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    fn temp_split_dir() -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "qwen-gguf-split-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    macro_rules! assert_rejects {
        ($path:expr, $pat:pat) => {{
            match GgufFile::open($path) {
                Ok(_) => panic!("expected rejection"),
                Err(e) => assert!(matches!(e, $pat), "got unexpected error: {e:?}"),
            }
        }};
    }

    #[test]
    fn minimal_gguf_round_trip() {
        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let g = GgufFile::open(&path).expect("minimal gguf should parse");
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].name, "t");
        assert_eq!(g.tensors[0].shard_idx, 0);
        assert_eq!(g.shard_count(), 1);
        let stamps = g
            .revalidate_retained_shard_stamps()
            .expect("retained shard stamp");
        assert_eq!(stamps.len(), 1);
        assert_eq!(stamps[0].shard_idx, 0);
        assert_eq!(stamps[0].path, path);
        assert_eq!(stamps[0].size, bytes.len() as u64);
        let retained = g.retained_shard_mmap(0).expect("retained shard");
        drop(g);
        assert_eq!(&retained[..4], &GGUF_MAGIC.to_le_bytes());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opened_file_constructor_uses_supplied_file_identity() {
        use std::os::unix::fs::MetadataExt;

        let original = build_minimal_gguf();
        let path = write_temp(&original);
        let moved = path.with_extension("opened-original.gguf");
        let file = File::open(&path).unwrap();
        let opened_inode = file.metadata().unwrap().ino();
        std::fs::rename(&path, &moved).unwrap();
        write_file(
            &path,
            &build_test_gguf(
                &[],
                &[TestTensor {
                    name: "replacement",
                    value: 2.0,
                }],
            ),
        );

        let gguf = GgufFile::from_opened_file(file, &path).unwrap();
        assert_eq!(gguf.tensors[0].name, "t");
        assert_eq!(
            gguf.revalidate_retained_shard_stamps().unwrap()[0].inode,
            opened_inode
        );
        assert_ne!(std::fs::metadata(&path).unwrap().ino(), opened_inode);

        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(moved).unwrap();
    }

    #[test]
    fn opened_file_constructor_rewinds_shared_cursor_without_reopening_path() {
        use std::io::{Seek, SeekFrom};

        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let moved = path.with_extension("opened-retained.gguf");
        let file = File::open(&path).unwrap();
        let mut shared_cursor = file.try_clone().unwrap();
        assert_eq!(
            shared_cursor.seek(SeekFrom::End(0)).unwrap(),
            bytes.len() as u64
        );
        std::fs::rename(&path, &moved).unwrap();
        assert!(!path.exists());

        let gguf = GgufFile::from_opened_file(file, &path).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "t");
        assert_eq!(gguf.shard_count(), 1);
        assert_eq!(&gguf.primary_shard().mmap[..4], &GGUF_MAGIC.to_le_bytes());

        std::fs::remove_file(moved).unwrap();
    }

    #[test]
    fn opened_file_constructor_rejects_split_metadata() {
        let path = write_temp(&build_split_shard(
            0,
            2,
            1,
            &[TestTensor {
                name: "t",
                value: 1.0,
            }],
        ));
        let error = GgufFile::from_opened_file(File::open(&path).unwrap(), &path)
            .err()
            .expect("split metadata must be rejected");
        assert!(error.to_string().contains("requires one shard"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_shard_read_keeps_opened_file_identity() {
        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let replacement = path.with_extension("replacement.gguf");
        let g = GgufFile::open(&path).expect("minimal gguf should parse");
        let tensor = g.find("t").expect("tensor").clone();
        g.revalidate_retained_shard_stamps()
            .expect("original retained stamp");

        write_file(&replacement, &vec![0xA5; bytes.len()]);
        std::fs::rename(&replacement, &path).expect("replace path");

        let mut actual = vec![0u8; tensor.n_bytes as usize];
        g.read_shard_exact_at(tensor.shard_idx, tensor.data_offset, &mut actual)
            .expect("read retained descriptor");
        assert_eq!(actual, g.try_slice(&tensor).expect("mapped tensor"));
        assert!(actual.iter().any(|&byte| byte != 0xA5));
        assert!(matches!(
            g.revalidate_retained_shard_stamps()
                .expect_err("path replacement changes retained inode ctime"),
            GgufError::Decode(_)
        ));

        assert!(matches!(
            g.read_shard_exact_at(99, 0, &mut actual),
            Err(GgufError::Decode(_))
        ));
        assert!(matches!(
            g.read_shard_exact_at(0, u64::MAX, &mut actual),
            Err(GgufError::Decode(_))
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn retained_shard_stamp_rejects_descriptor_truncation() {
        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let g = GgufFile::open(&path).expect("minimal gguf should parse");

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open source for truncation")
            .set_len((bytes.len() - 1) as u64)
            .expect("truncate source");
        assert!(matches!(
            g.revalidate_retained_shard_stamps(),
            Err(GgufError::Decode(_))
        ));

        drop(g);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn opens_split_gguf_from_first_shard() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                2,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );

        let g = GgufFile::open(&first).expect("split gguf should parse");
        assert_eq!(g.shard_count(), 2);
        let stamps = g
            .revalidate_retained_shard_stamps()
            .expect("split retained stamps");
        assert_eq!(stamps.len(), 2);
        assert_eq!(stamps[0].shard_idx, 0);
        assert_eq!(stamps[0].path, first);
        assert_eq!(stamps[1].shard_idx, 1);
        assert_eq!(stamps[1].path, second);
        assert_eq!(g.tensors.len(), 2);
        let a = g.find("a").expect("a tensor");
        let b = g.find("b").expect("b tensor");
        assert_eq!(a.shard_idx, 0);
        assert_eq!(b.shard_idx, 1);
        assert_eq!(
            a.data_offset, b.data_offset,
            "shard-local offsets may match"
        );
        assert_eq!(g.slice(a), &1.0f32.to_le_bytes());
        assert_eq!(g.slice(b), &2.0f32.to_le_bytes());
        assert_eq!(
            g.try_shard_range(a.shard_idx, a.data_offset, 4)
                .expect("validated shard range"),
            &1.0f32.to_le_bytes()
        );
        let mut exact = [0u8; 4];
        g.read_shard_exact_at(a.shard_idx, a.data_offset, &mut exact)
            .expect("exact retained-descriptor read");
        assert_eq!(exact, 1.0f32.to_le_bytes());
        g.read_shard_exact_at(b.shard_idx, b.data_offset, &mut exact)
            .expect("exact second-shard read");
        assert_eq!(exact, 2.0f32.to_le_bytes());

        let mut forged = a.clone();
        forged.shard_idx = 99;
        assert!(matches!(g.try_slice(&forged), Err(GgufError::Decode(_))));

        let mut forged = b.clone();
        forged.data_offset = u64::MAX;
        assert!(matches!(g.try_slice(&forged), Err(GgufError::Decode(_))));
        assert!(matches!(
            g.try_shard_range(99, 0, 4),
            Err(GgufError::Decode(_))
        ));
        assert!(matches!(
            g.try_shard_range(0, u64::MAX, 4),
            Err(GgufError::Decode(_))
        ));
        let first_len = g.shard_mapped_lengths()[0];
        assert_eq!(
            g.try_shard_range(0, first_len as u64, 0)
                .expect("exact-end empty range"),
            &[] as &[u8]
        );
        assert!(matches!(
            g.try_shard_range(0, (first_len - 1) as u64, 2),
            Err(GgufError::Decode(_))
        ));
        let mut empty = [];
        g.read_shard_exact_at(0, first_len as u64, &mut empty)
            .expect("exact-end empty read");
        assert!(matches!(
            g.read_shard_exact_at(0, (first_len - 1) as u64, &mut exact[..2]),
            Err(GgufError::Decode(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn opens_split_gguf_with_unpadded_empty_first_shard() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(&first, &build_unpadded_empty_split_shard(0, 2, 1));
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                1,
                &[TestTensor {
                    name: "only",
                    value: 3.0,
                }],
            ),
        );

        let gguf = GgufFile::open(&first).expect("metadata-only first shard should parse");
        assert_eq!(gguf.shard_count(), 2);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(
            gguf.slice(gguf.find("only").unwrap()),
            &3.0f32.to_le_bytes()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unpadded_shard_exception_rejects_partial_padding_and_tensor_tables() {
        let mut partially_padded = build_unpadded_empty_split_shard(0, 2, 1);
        partially_padded.push(0);
        let partial_path = write_temp(&partially_padded);
        let partial_error =
            open_one_shard_file(File::open(&partial_path).unwrap(), &partial_path, 0)
                .err()
                .expect("partial empty-shard padding must be rejected");
        assert!(partial_error.to_string().contains("aligned header end"));

        let nonempty_path = write_temp(&build_unpadded_nonempty_gguf());
        let nonempty_error =
            open_one_shard_file(File::open(&nonempty_path).unwrap(), &nonempty_path, 0)
                .err()
                .expect("nonempty unpadded shard must be rejected");
        assert!(!nonempty_error.to_string().is_empty());

        let _ = std::fs::remove_file(partial_path);
        let _ = std::fs::remove_file(nonempty_path);
    }

    #[test]
    fn rejects_opening_nonfirst_split() {
        let dir = temp_split_dir();
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                2,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&second, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_missing_split_sibling() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Io(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_bad_split_suffix() {
        let dir = temp_split_dir();
        let first = dir.join("model.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                1,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_split_no() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_split_count() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                1,
                3,
                2,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_duplicate_tensor_names_across_splits() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                2,
                &[TestTensor {
                    name: "dup",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                2,
                &[TestTensor {
                    name: "dup",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_split_tensor_count_mismatch() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                3,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                3,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_split_tensor_count_overrun_while_merging() {
        let dir = temp_split_dir();
        let first = dir.join("model-00001-of-00002.gguf");
        let second = dir.join("model-00002-of-00002.gguf");
        write_file(
            &first,
            &build_split_shard(
                0,
                2,
                1,
                &[TestTensor {
                    name: "a",
                    value: 1.0,
                }],
            ),
        );
        write_file(
            &second,
            &build_split_shard(
                1,
                2,
                1,
                &[TestTensor {
                    name: "b",
                    value: 2.0,
                }],
            ),
        );
        assert_rejects!(&first, GgufError::Decode(_));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_absurd_split_count_before_allocating_paths() {
        let path = write_temp(&build_test_gguf(
            &[
                TestKv::U16(SPLIT_NO_KEY, 0),
                TestKv::U64(SPLIT_COUNT_KEY, MAX_SPLITS + 1),
                TestKv::I32(SPLIT_TENSORS_COUNT_KEY, 1),
            ],
            &[TestTensor {
                name: "a",
                value: 1.0,
            }],
        ));
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_absurd_split_tensor_count_before_allocating_table() {
        let path = write_temp(&build_test_gguf(
            &[
                TestKv::U16(SPLIT_NO_KEY, 0),
                TestKv::U16(SPLIT_COUNT_KEY, 2),
                TestKv::U64(SPLIT_TENSORS_COUNT_KEY, MAX_TENSORS + 1),
            ],
            &[TestTensor {
                name: "a",
                value: 1.0,
            }],
        ));
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_overlap_inside_one_shard() {
        let path = write_temp(&build_overlapping_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_tensor_rank_before_parser_can_panic() {
        let path = write_temp(&build_bad_rank_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_tensor_dimension_before_parser_can_panic() {
        let path = write_temp(&build_bad_dimension_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_tensor_type_before_parser_can_panic() {
        let path = write_temp(&build_bad_type_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_quant_block_alignment_before_slicing() {
        let path = write_temp(&build_bad_quant_alignment_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_nested_metadata_array_before_recursing() {
        let path = write_temp(&build_nested_array_metadata_gguf());
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_minimal_gguf();
        bytes[0] ^= 0xFF;
        let path = write_temp(&bytes);
        assert_rejects!(&path, GgufError::BadMagic);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = build_minimal_gguf();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let path = write_temp(&bytes);
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_truncated_header() {
        let bytes = vec![0x47u8, 0x47, 0x55, 0x46];
        let path = write_temp(&bytes);
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_absurd_tensor_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let path = write_temp(&bytes);
        assert_rejects!(&path, GgufError::Decode(_));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn finds_token_embedding() {
        let path = fixture();
        let g = GgufFile::open(&path).expect("open");
        // Every model has these (or token_embd.weight in llama-style names).
        let candidates = ["token_embd.weight", "tok_embeddings.weight"];
        let found = candidates.iter().any(|n| g.find(n).is_some());
        if !found {
            // Print what we did find, to make failure actionable.
            for t in g.tensors.iter().take(5) {
                eprintln!("  {} {:?} {:?}", t.name, t.shape, t.dtype);
            }
        }
        assert!(found, "no token embedding tensor found");
    }

    #[test]
    fn stop_token_ids_on_real_gguf() {
        let path = fixture();
        let g = GgufFile::open(&path).expect("open");
        let stops = g
            .stop_token_ids()
            .expect("real Qwen 3.5 GGUF must declare a stop token");
        assert!(!stops.is_empty());
        // The smallest local fixture is the 0.8B instruct GGUF; if both
        // 0.8B and the bundled gguf-rs LE-v3 fixture are unavailable we
        // wouldn't have reached this assertion. For instruct Qwen 3.5
        // the declared EOS is 248046.
        if path.to_string_lossy().contains("Qwen3.5-0.8B") {
            assert_eq!(stops, vec![248046], "0.8B instruct EOS");
        }
    }

    #[test]
    fn stop_token_ids_missing_key_errors() {
        // Use the minimal-gguf fixture (no metadata at all) to exercise
        // the missing-key path without depending on a real model file.
        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let g = GgufFile::open(&path).expect("minimal gguf opens");
        let err = g
            .stop_token_ids()
            .expect_err("minimal GGUF declares no EOS");
        assert!(
            matches!(err, GgufError::MissingKey("tokenizer.ggml.eos_token_id")),
            "expected MissingKey, got {err:?}",
        );
        let _ = std::fs::remove_file(&path);
    }
}
