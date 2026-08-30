use anyhow::{Context, Result, ensure};
use blake3::Hasher as Blake3Hasher;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use zip::{CompressionMethod, ZipArchive};

const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct ArchiveSpec<'a> {
    pub(crate) root: &'a str,
    pub(crate) layer_count: usize,
    pub(crate) hidden_size: usize,
    pub(crate) matrix_bytes: u64,
    pub(crate) data_pickle_sha256: &'a str,
    pub(crate) serialization_id: Option<&'a str>,
    pub(crate) identity_storage_index: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedMatrix {
    pub(crate) storage_index: usize,
    pub(crate) byte_offset: u64,
    pub(crate) byte_length: u64,
    pub(crate) blake3: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedPayload {
    pub(crate) byte_length: u64,
    pub(crate) blake3: String,
    pub(crate) matrices: Vec<ExtractedMatrix>,
}

pub(crate) fn validate_archive<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    spec: ArchiveSpec<'_>,
) -> Result<()> {
    let expected_matrix_bytes = (spec.hidden_size as u64)
        .checked_mul(spec.hidden_size as u64)
        .and_then(|words| words.checked_mul(2))
        .context("pinned matrix byte count overflow")?;
    ensure!(
        spec.matrix_bytes == expected_matrix_bytes,
        "pinned matrix byte count does not match hidden size"
    );
    ensure!(
        spec.identity_storage_index
            .is_none_or(|index| index < spec.layer_count),
        "identity storage index is outside the archive layer inventory"
    );
    let expected_names = expected_archive_names(spec);
    ensure!(
        archive.len() == expected_names.len(),
        "pinned torch ZIP has {} entries; expected {}",
        archive.len(),
        expected_names.len()
    );
    let mut actual_names = BTreeSet::new();
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .with_context(|| format!("inspect pinned torch ZIP entry {index}"))?;
        ensure!(
            !entry.is_dir(),
            "pinned torch ZIP contains a directory entry"
        );
        let name = entry.name().to_owned();
        ensure!(
            entry
                .enclosed_name()
                .is_some_and(|path| path == Path::new(&name)),
            "pinned torch ZIP contains unsafe path {name:?}"
        );
        ensure!(
            actual_names.insert(name.clone()),
            "pinned torch ZIP contains duplicate entry {name:?}"
        );
        ensure!(
            entry.compression() == CompressionMethod::Stored
                && entry.compressed_size() == entry.size(),
            "pinned torch ZIP entry {name:?} is not stored verbatim"
        );
    }
    ensure!(
        actual_names == expected_names,
        "pinned torch ZIP entry inventory is not canonical"
    );

    let pickle_name = format!("{}/data.pkl", spec.root);
    let mut pickle = archive
        .by_name(&pickle_name)
        .context("open pinned data.pkl")?;
    ensure!(pickle.size() <= 16 * 1024, "pinned data.pkl exceeds limit");
    let mut pickle_bytes = Vec::new();
    pickle
        .read_to_end(&mut pickle_bytes)
        .context("read pinned data.pkl")?;
    let pickle_digest = super::hex(&Sha256::digest(&pickle_bytes));
    ensure!(
        pickle_digest == spec.data_pickle_sha256,
        "pinned data.pkl SHA-256 mismatch"
    );
    drop(pickle);

    for (name, expected) in [
        (format!("{}/.format_version", spec.root), "1"),
        (format!("{}/.storage_alignment", spec.root), "64"),
        (format!("{}/byteorder", spec.root), "little"),
        (format!("{}/version", spec.root), "3"),
    ] {
        validate_text_entry(archive, &name, expected)?;
    }
    if let Some(expected) = spec.serialization_id {
        validate_text_entry(
            archive,
            &format!("{}/.data/serialization_id", spec.root),
            expected,
        )?;
    }
    for layer in 0..spec.layer_count {
        let name = format!("{}/data/{layer}", spec.root);
        let entry = archive
            .by_name(&name)
            .with_context(|| format!("open pinned storage layer {layer}"))?;
        ensure!(
            entry.size() == spec.matrix_bytes,
            "pinned storage layer {layer} length {} != expected {}",
            entry.size(),
            spec.matrix_bytes
        );
    }
    Ok(())
}

fn validate_text_entry<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
    expected: &str,
) -> Result<()> {
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("open pinned metadata {name}"))?;
    ensure!(entry.size() <= 128, "pinned metadata {name} exceeds limit");
    let mut value = String::new();
    entry
        .read_to_string(&mut value)
        .with_context(|| format!("read pinned metadata {name}"))?;
    ensure!(value.trim() == expected, "pinned metadata {name} mismatch");
    Ok(())
}

fn expected_archive_names(spec: ArchiveSpec<'_>) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for suffix in [
        "data.pkl",
        ".format_version",
        ".storage_alignment",
        "byteorder",
        "version",
        ".data/serialization_id",
    ] {
        names.insert(format!("{}/{suffix}", spec.root));
    }
    for layer in 0..spec.layer_count {
        names.insert(format!("{}/data/{layer}", spec.root));
    }
    names
}

pub(crate) fn extract_payload<R: Read + Seek, W: Write>(
    archive: &mut ZipArchive<R>,
    spec: ArchiveSpec<'_>,
    output: &mut W,
) -> Result<ExtractedPayload> {
    let mut hasher = Blake3Hasher::new();
    let buffer_length = usize::try_from(spec.matrix_bytes.min(COPY_BUFFER_BYTES as u64))
        .context("F16 copy buffer length")?;
    let mut buffer = vec![0u8; buffer_length];
    ensure!(
        !buffer.is_empty() && buffer.len().is_multiple_of(2),
        "copy buffer must preserve F16 words"
    );
    let mut byte_length = 0u64;
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(spec.layer_count)
        .context("allocate extracted matrix descriptors")?;
    for layer in 0..spec.layer_count {
        let name = format!("{}/data/{layer}", spec.root);
        let mut entry = archive
            .by_name(&name)
            .with_context(|| format!("open pinned storage layer {layer}"))?;
        let matrix_offset = byte_length;
        let mut matrix_hasher = Blake3Hasher::new();
        let mut remaining = spec.matrix_bytes;
        while remaining > 0 {
            let matrix_byte_offset = spec.matrix_bytes - remaining;
            let chunk_length = usize::try_from(remaining.min(buffer.len() as u64))
                .context("F16 copy chunk length")?;
            entry
                .read_exact(&mut buffer[..chunk_length])
                .with_context(|| format!("read pinned storage layer {layer}"))?;
            ensure_finite_f16(
                &buffer[..chunk_length],
                layer,
                matrix_byte_offset as usize / 2,
            )?;
            if spec.identity_storage_index == Some(layer) {
                ensure_identity_f16(
                    &buffer[..chunk_length],
                    matrix_byte_offset as usize / 2,
                    spec.hidden_size,
                    layer,
                )?;
            }
            output
                .write_all(&buffer[..chunk_length])
                .with_context(|| format!("write imported storage layer {layer}"))?;
            hasher.update(&buffer[..chunk_length]);
            matrix_hasher.update(&buffer[..chunk_length]);
            remaining -= chunk_length as u64;
            byte_length = byte_length
                .checked_add(chunk_length as u64)
                .context("imported payload length overflow")?;
        }
        let mut extra = [0u8; 1];
        ensure!(
            entry
                .read(&mut extra)
                .with_context(|| format!("check pinned storage layer {layer} end"))?
                == 0,
            "pinned storage layer {layer} contains trailing bytes"
        );
        matrices.push(ExtractedMatrix {
            storage_index: layer,
            byte_offset: matrix_offset,
            byte_length: spec.matrix_bytes,
            blake3: matrix_hasher.finalize().to_hex().to_string(),
        });
    }
    let expected_bytes = spec
        .matrix_bytes
        .checked_mul(spec.layer_count as u64)
        .context("expected payload byte length overflow")?;
    ensure!(
        byte_length == expected_bytes,
        "imported payload length {byte_length} != expected {expected_bytes}"
    );
    Ok(ExtractedPayload {
        byte_length,
        blake3: hasher.finalize().to_hex().to_string(),
        matrices,
    })
}

fn ensure_identity_f16(
    bytes: &[u8],
    word_offset: usize,
    hidden_size: usize,
    layer: usize,
) -> Result<()> {
    ensure!(hidden_size > 0, "identity matrix hidden size is zero");
    let (words, remainder) = bytes.as_chunks::<2>();
    ensure!(remainder.is_empty(), "identity F16 chunk has trailing byte");
    for (index, chunk) in words.iter().enumerate() {
        let coordinate = word_offset
            .checked_add(index)
            .context("identity matrix coordinate overflow")?;
        let row = coordinate / hidden_size;
        let column = coordinate % hidden_size;
        let bits = u16::from_le_bytes(*chunk);
        let valid = if row == column {
            bits == 0x3c00
        } else {
            bits & 0x7fff == 0
        };
        ensure!(
            valid,
            "pinned storage layer {layer} is not the claimed F16 identity at row {row}, column {column}"
        );
    }
    Ok(())
}

pub(crate) fn ensure_finite_f16(bytes: &[u8], layer: usize, word_offset: usize) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "F16 chunk has odd byte length"
    );
    let (words, remainder) = bytes.as_chunks::<2>();
    ensure!(remainder.is_empty(), "F16 chunk has trailing byte");
    for (index, chunk) in words.iter().enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        ensure!(
            bits & 0x7c00 != 0x7c00,
            "pinned storage layer {layer} contains non-finite F16 at matrix word {}",
            word_offset + index
        );
    }
    Ok(())
}

pub(crate) fn hash_sha256(file: &mut File, path: &Path) -> Result<String> {
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind {} for SHA-256", path.display()))?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_BYTES, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(super::hex(&hasher.finalize()))
}
