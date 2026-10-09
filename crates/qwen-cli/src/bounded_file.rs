//! Descriptor-based, bounded regular-file reads; limits and content hashes belong
//! to callers. This is not ancestor confinement or an immutable file snapshot.

use anyhow::{Context, Result, ensure};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub fn open_regular_file(path: &Path) -> Result<(File, usize)> {
    let lexical_metadata =
        std::fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        lexical_metadata.file_type().is_file() && !lexical_metadata.file_type().is_symlink(),
        "{} must be a regular non-symlink file",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        // A leaf can change type between inspection and open. Do not block on
        // that substituted file before validating the opened descriptor below.
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect opened {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "{} must remain a regular file after open",
        path.display()
    );
    let length = usize::try_from(metadata.len())
        .with_context(|| format!("{} length does not fit this platform", path.display()))?;
    Ok((file, length))
}

/// Reads from the current offset without reopening or seeking. Rejects a short
/// read or trailing bytes, not same-length edits made through another descriptor.
pub fn read_opened_file_exact(
    mut file: File,
    path: &Path,
    expected_length: usize,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_length)
        .with_context(|| format!("allocate {} bytes for {}", expected_length, path.display()))?;
    bytes.resize(expected_length, 0);
    file.read_exact(&mut bytes)
        .with_context(|| format!("read exact contents of {}", path.display()))?;
    let mut extra = [0u8; 1];
    ensure!(
        file.read(&mut extra)
            .with_context(|| format!("check end of {}", path.display()))?
            == 0,
        "{} grew while it was being read",
        path.display()
    );
    Ok(bytes)
}

pub fn read_regular_file_exact(path: &Path, expected_length: usize) -> Result<Vec<u8>> {
    let (file, length) = open_regular_file(path)?;
    ensure!(
        length == expected_length,
        "{} length {} != expected {}",
        path.display(),
        length,
        expected_length
    );
    read_opened_file_exact(file, path, expected_length)
}

pub fn read_regular_file_bounded(path: &Path, maximum_length: usize) -> Result<Vec<u8>> {
    let (file, length) = open_regular_file(path)?;
    ensure!(
        length <= maximum_length,
        "{} length {} exceeds limit {}",
        path.display(),
        length,
        maximum_length
    );
    read_opened_file_exact(file, path, length)
}

#[cfg(test)]
mod tests;
