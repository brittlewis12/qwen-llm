use anyhow::{Context, Result, ensure};
use blake3::Hasher as Blake3Hasher;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use zip::{CompressionMethod, ZipArchive};

const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum ArchiveLayout {
    LayerStorages,
    ContiguousStorage { storage_index: usize },
}

impl ArchiveLayout {
    pub(crate) const fn storage_index_for_layer(self, layer: usize) -> usize {
        match self {
            Self::LayerStorages => layer,
            Self::ContiguousStorage { storage_index } => storage_index,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ArchiveSpec<'a> {
    pub(crate) root: &'a str,
    pub(crate) layout: ArchiveLayout,
    pub(crate) layer_count: usize,
    pub(crate) hidden_size: usize,
    pub(crate) matrix_bytes: u64,
    pub(crate) data_pickle_sha256: &'a str,
    pub(crate) serialization_id: Option<&'a str>,
    pub(crate) identity_layer_index: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExtractedMatrix {
    pub(crate) archive_storage_index: usize,
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
        spec.identity_layer_index
            .is_none_or(|index| index < spec.layer_count),
        "identity layer index is outside the archive layer inventory"
    );
    let payload_bytes = spec
        .matrix_bytes
        .checked_mul(spec.layer_count as u64)
        .context("pinned payload byte count overflow")?;
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
    match spec.layout {
        ArchiveLayout::LayerStorages => {
            for layer in 0..spec.layer_count {
                validate_storage_length(archive, spec, layer, spec.matrix_bytes)?;
            }
        }
        ArchiveLayout::ContiguousStorage { storage_index } => {
            validate_storage_length(archive, spec, storage_index, payload_bytes)?;
        }
    }
    Ok(())
}

fn validate_storage_length<R: Read + Seek>(
    archive: &mut ZipArchive<R>,
    spec: ArchiveSpec<'_>,
    storage_index: usize,
    expected_bytes: u64,
) -> Result<()> {
    let name = format!("{}/data/{storage_index}", spec.root);
    let entry = archive
        .by_name(&name)
        .with_context(|| format!("open pinned storage {storage_index}"))?;
    ensure!(
        entry.size() == expected_bytes,
        "pinned storage {storage_index} length {} != expected {expected_bytes}",
        entry.size()
    );
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
    match spec.layout {
        ArchiveLayout::LayerStorages => {
            for layer in 0..spec.layer_count {
                names.insert(format!("{}/data/{layer}", spec.root));
            }
        }
        ArchiveLayout::ContiguousStorage { storage_index } => {
            names.insert(format!("{}/data/{storage_index}", spec.root));
        }
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
    match spec.layout {
        ArchiveLayout::LayerStorages => {
            for layer in 0..spec.layer_count {
                let storage_index = spec.layout.storage_index_for_layer(layer);
                let name = format!("{}/data/{storage_index}", spec.root);
                let mut entry = archive
                    .by_name(&name)
                    .with_context(|| format!("open pinned storage {storage_index}"))?;
                matrices.push(copy_matrix(
                    &mut entry,
                    output,
                    &mut hasher,
                    &mut buffer,
                    spec,
                    layer,
                    storage_index,
                    &mut byte_length,
                )?);
                ensure_storage_end(&mut entry, storage_index)?;
            }
        }
        ArchiveLayout::ContiguousStorage { storage_index } => {
            let name = format!("{}/data/{storage_index}", spec.root);
            let mut entry = archive
                .by_name(&name)
                .with_context(|| format!("open pinned storage {storage_index}"))?;
            for layer in 0..spec.layer_count {
                matrices.push(copy_matrix(
                    &mut entry,
                    output,
                    &mut hasher,
                    &mut buffer,
                    spec,
                    layer,
                    storage_index,
                    &mut byte_length,
                )?);
            }
            ensure_storage_end(&mut entry, storage_index)?;
        }
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

#[allow(clippy::too_many_arguments)]
fn copy_matrix<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    whole_hasher: &mut Blake3Hasher,
    buffer: &mut [u8],
    spec: ArchiveSpec<'_>,
    layer: usize,
    storage_index: usize,
    byte_length: &mut u64,
) -> Result<ExtractedMatrix> {
    let matrix_offset = *byte_length;
    let mut matrix_hasher = Blake3Hasher::new();
    let mut remaining = spec.matrix_bytes;
    while remaining > 0 {
        let matrix_byte_offset = spec.matrix_bytes - remaining;
        let chunk_length =
            usize::try_from(remaining.min(buffer.len() as u64)).context("F16 copy chunk length")?;
        input
            .read_exact(&mut buffer[..chunk_length])
            .with_context(|| format!("read pinned transport layer {layer}"))?;
        ensure_finite_f16(
            &buffer[..chunk_length],
            layer,
            matrix_byte_offset as usize / 2,
        )?;
        if spec.identity_layer_index == Some(layer) {
            ensure_identity_f16(
                &buffer[..chunk_length],
                matrix_byte_offset as usize / 2,
                spec.hidden_size,
                layer,
            )?;
        }
        output
            .write_all(&buffer[..chunk_length])
            .with_context(|| format!("write imported transport layer {layer}"))?;
        whole_hasher.update(&buffer[..chunk_length]);
        matrix_hasher.update(&buffer[..chunk_length]);
        remaining -= chunk_length as u64;
        *byte_length = byte_length
            .checked_add(chunk_length as u64)
            .context("imported payload length overflow")?;
    }
    Ok(ExtractedMatrix {
        archive_storage_index: storage_index,
        byte_offset: matrix_offset,
        byte_length: spec.matrix_bytes,
        blake3: matrix_hasher.finalize().to_hex().to_string(),
    })
}

fn ensure_storage_end<R: Read>(input: &mut R, storage_index: usize) -> Result<()> {
    let mut extra = [0u8; 1];
    ensure!(
        input
            .read(&mut extra)
            .with_context(|| format!("check pinned storage {storage_index} end"))?
            == 0,
        "pinned storage {storage_index} contains trailing bytes"
    );
    Ok(())
}

pub(crate) fn ensure_identity_f16(
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
            bits == 0
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    fn f16_words(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn test_archive(storages: &[(usize, Vec<u8>)], data_pickle: &[u8]) -> Vec<u8> {
        let mut output = Cursor::new(Vec::new());
        {
            let mut writer = ZipWriter::new(&mut output);
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            for (name, bytes) in [
                ("lens/data.pkl", data_pickle),
                ("lens/.format_version", b"1" as &[u8]),
                ("lens/.storage_alignment", b"64" as &[u8]),
                ("lens/byteorder", b"little" as &[u8]),
                ("lens/version", b"3" as &[u8]),
                ("lens/.data/serialization_id", b"fixture" as &[u8]),
            ] {
                writer.start_file(name, options).unwrap();
                writer.write_all(bytes).unwrap();
            }
            for (storage_index, bytes) in storages {
                writer
                    .start_file(format!("lens/data/{storage_index}"), options)
                    .unwrap();
                writer.write_all(bytes).unwrap();
            }
            writer.finish().unwrap();
        }
        output.into_inner()
    }

    fn extract_fixture(
        storages: &[(usize, Vec<u8>)],
        layout: ArchiveLayout,
    ) -> Result<(Vec<u8>, ExtractedPayload)> {
        let pickle = b"opaque fixture; never interpreted";
        let digest = super::super::hex(&Sha256::digest(pickle));
        let spec = ArchiveSpec {
            root: "lens",
            layout,
            layer_count: 2,
            hidden_size: 2,
            matrix_bytes: 8,
            data_pickle_sha256: &digest,
            serialization_id: Some("fixture"),
            identity_layer_index: Some(1),
        };
        let bytes = test_archive(storages, pickle);
        let mut archive = ZipArchive::new(Cursor::new(bytes))?;
        validate_archive(&mut archive, spec)?;
        let mut output = Vec::new();
        let extracted = extract_payload(&mut archive, spec, &mut output)?;
        Ok((output, extracted))
    }

    #[test]
    fn extracts_separate_and_contiguous_storages_without_executing_pickle() {
        let first = f16_words(&[0x4000, 0, 0, 0x4000]);
        let identity = f16_words(&[0x3c00, 0, 0, 0x3c00]);
        let expected = [first.clone(), identity.clone()].concat();

        let (separate_bytes, separate) = extract_fixture(
            &[(1, identity.clone()), (0, first.clone())],
            ArchiveLayout::LayerStorages,
        )
        .unwrap();
        assert_eq!(separate_bytes, expected);
        assert_eq!(
            separate
                .matrices
                .iter()
                .map(|matrix| matrix.archive_storage_index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(separate.matrices[1].byte_offset, 8);

        let (contiguous_bytes, contiguous) = extract_fixture(
            &[(7, expected.clone())],
            ArchiveLayout::ContiguousStorage { storage_index: 7 },
        )
        .unwrap();
        assert_eq!(contiguous_bytes, expected);
        assert_eq!(
            contiguous
                .matrices
                .iter()
                .map(|matrix| matrix.archive_storage_index)
                .collect::<Vec<_>>(),
            [7, 7]
        );
        assert_eq!(contiguous.matrices[1].byte_offset, 8);
        assert_eq!(contiguous.blake3, separate.blake3);
    }

    #[test]
    fn contiguous_storage_still_rejects_nonfinite_layers() {
        let first = f16_words(&[0x4000, 0, 0, 0x4000]);
        let nonfinite = f16_words(&[0x7c00, 0, 0, 0x3c00]);
        let error = extract_fixture(
            &[(0, [first, nonfinite].concat())],
            ArchiveLayout::ContiguousStorage { storage_index: 0 },
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-finite F16"));
    }
}
