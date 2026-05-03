//! GGUF v3 mmap-only loader.
//!
//! Parsing is delegated to `gguf-rs` (Britt's own crate, in `~/code/gguf`)
//! for metadata + tensor-table semantics. We layer:
//!
//! 1. An independent `Mmap` of the file so kernels read tensor bytes
//!    directly out of the page cache. **No tensor data is ever copied at
//!    load time.**
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

use crate::tensor::{GgmlType, TensorDesc};
use gguf_rs::{ByteOrder as GByteOrder, GGUFContainer, GGUFModel};
use memmap2::Mmap;
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

/// "GGUF" in little-endian (only LE is supported; see [`open`] preconditions).
const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
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

/// One opened GGUF file.
///
/// * `mmap` is held for the full lifetime; tensor slices reference it.
/// * `tensors` are absolute-offset descriptors (origin = start of file).
/// * `model` is the gguf-rs decoded view of metadata + tensor table.
#[allow(dead_code)] // Debug is used by tests via expect_err
pub struct GgufFile {
    pub mmap: Mmap,
    pub tensors: Vec<TensorDesc>,
    pub model: GGUFModel,
    /// Absolute byte offset of the start of the tensor-data section.
    /// All `TensorDesc.data_offset` values include this.
    pub tensor_data_start: u64,
    /// Tensor-data alignment in bytes (from `general.alignment`, default 32).
    pub alignment: u64,
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

        // Open + mmap the file. mmap is the source of truth for tensor data.
        let file = File::open(path)?;
        // SAFETY: regular file held for the lifetime of `Self`. Memory
        // mapping a file handed to us by the user is the standard path; if
        // the file is concurrently truncated underneath us we'll SIGBUS on
        // access — that is an OS-level signal we cannot prevent in safe
        // Rust without copying, and would be the user racing themselves.
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate magic against the mmap directly. The mmap is the *only*
        // path that the rest of this function trusts; the streaming parser
        // is given a separate `File` and its results are reconciled below.
        validate_magic(&mmap)?;
        let version = read_u32_at(&mmap, 4)?;
        if version != 3 {
            return Err(GgufError::Decode(format!(
                "unsupported GGUF version {version} (only v3 supported)"
            )));
        }

        // Hand gguf-rs a fresh, separate File, positioned just past the
        // magic (its `decode()` reads version next, not magic). gguf-rs is
        // the metadata parser of record; we re-derive structural offsets
        // independently below.
        let mut parse_file = File::open(path)?;
        parse_file.seek(SeekFrom::Start(4))?;
        let mut container = GGUFContainer::new(
            GByteOrder::LE,
            Box::new(BufReader::with_capacity(64 * 1024, parse_file)),
            u64::MAX,
        );
        let model = container.decode()?;

        // Read and validate alignment.
        let alignment = read_alignment(&model)?;

        // Independently locate tensor-data start by replaying the header
        // layout against the mmap. This both validates the on-disk
        // structure and yields the offset we need for absolute tensor
        // addressing. Cross-checks the (num_tensors, num_kv) against
        // gguf-rs's parse to catch any disagreement.
        let tensor_data_start = locate_tensor_data_start(&mmap, &model, alignment)?;

        // Build TensorDesc list with absolute offsets, validating every
        // tensor lies fully inside the file with no overflow, with no
        // overlap between any two tensor data ranges.
        //
        // Invariants enforced (cf. gguf-validator SPEC.md):
        // * TENSOR-06: every dim > 0
        // * TENSOR-07: every dim <= MAX_DIMENSION
        // * TENSOR-08/09: product fits in u64 and <= MAX_ELEMENTS
        // * TENSOR-13: tensor offset is a multiple of alignment
        // * LAYOUT-02/05/08: tensor range fits in file with no overflow
        // * LAYOUT-03: no two tensor ranges overlap
        let mmap_len = mmap.len() as u64;
        let mut tensors: Vec<TensorDesc> = Vec::with_capacity(model.tensors().len());
        for t in model.tensors() {
            // gguf-rs stores trailing 1s for unused dims; drop them so
            // `shape.len()` reflects actual rank.
            let mut shape: Vec<u64> = t.shape.iter().copied().collect();
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

            // TENSOR-13: t.offset is relative to data start; alignment
            // applies to that. (Equivalent to data_offset % alignment == 0
            // since tensor_data_start itself is aligned.)
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
                data_offset,
                n_bytes: t.size,
            });
        }

        // LAYOUT-03: no two tensor data ranges overlap. Sort indices by
        // offset, then check adjacent pairs.
        let mut sort_idx: Vec<usize> = (0..tensors.len()).collect();
        sort_idx.sort_by_key(|&i| tensors[i].data_offset);
        for w in sort_idx.windows(2) {
            let (a, b) = (&tensors[w[0]], &tensors[w[1]]);
            if a.data_offset + a.n_bytes > b.data_offset {
                return Err(GgufError::Decode(format!(
                    "tensors {:?} and {:?} have overlapping data ranges",
                    a.name, b.name
                )));
            }
        }

        Ok(Self {
            mmap,
            tensors,
            model,
            tensor_data_start,
            alignment,
        })
    }

    /// Slice into the mmap for `desc`. Slice lifetime is tied to `&self`.
    pub fn slice(&self, desc: &TensorDesc) -> &[u8] {
        let start = desc.data_offset as usize;
        let end = start + desc.n_bytes as usize;
        &self.mmap[start..end]
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
        value_as_u64(self.model.metadata().get(key))
    }

    /// Convenience: lookup a string-typed metadata value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.model.metadata().get(key).and_then(|v| v.as_str())
    }
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
    v.and_then(|v| v.as_u64())
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
        4 | 5 | 6 => {
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
        10 | 11 | 12 => {
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
        assert!(g.tensors.len() > 0, "tensor table is empty");
        assert!(g.tensor_data_start > 0);
        // tensor data start must be aligned.
        assert_eq!(g.tensor_data_start % g.alignment, 0);

        // First tensor must begin exactly at tensor_data_start
        // (the smallest relative offset is always 0).
        let min_offset = g.tensors.iter().map(|t| t.data_offset).min().unwrap();
        assert_eq!(min_offset, g.tensor_data_start);

        // Last tensor must end inside the file.
        let max_end = g
            .tensors
            .iter()
            .map(|t| t.data_offset + t.n_bytes)
            .max()
            .unwrap() as usize;
        assert!(
            max_end <= g.mmap.len(),
            "max tensor end {} > file size {}",
            max_end,
            g.mmap.len()
        );

        eprintln!(
            "[gguf-test] {}: arch={:?}, {} tensors, data starts at {}, file is {} MiB",
            path.display(),
            g.architecture(),
            g.tensors.len(),
            g.tensor_data_start,
            g.mmap.len() / (1024 * 1024),
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

    #[test]
    fn minimal_gguf_round_trip() {
        let bytes = build_minimal_gguf();
        let path = write_temp(&bytes);
        let g = GgufFile::open(&path).expect("minimal gguf should parse");
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].name, "t");
        let _ = std::fs::remove_file(&path);
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
}
