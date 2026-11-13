use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

pub(crate) fn hash_key_part(hasher: &mut Sha256, label: &str, value: &[u8]) {
    hasher.update((label.len() as u64).to_be_bytes());
    hasher.update(label.as_bytes());
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

pub(crate) fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").unwrap();
    }
    out
}

pub(crate) struct SourceCacheKey {
    pub hash: String,
    pub dependencies: Vec<PathBuf>,
}

fn hash_source_tree(
    path: &Path,
    label: &str,
    hasher: &mut Sha256,
    seen: &mut BTreeSet<PathBuf>,
    dependencies: &mut Vec<PathBuf>,
) -> io::Result<()> {
    if !seen.insert(path.canonicalize()?) {
        return Ok(());
    }
    let bytes = std::fs::read(path)?;
    hash_key_part(hasher, label, &bytes);
    dependencies.push(path.to_path_buf());
    let source = std::str::from_utf8(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    for line in source.lines() {
        let Some(directive) = line.trim_start().strip_prefix('#') else {
            continue;
        };
        let Some(include) = directive.trim_start().strip_prefix("include") else {
            continue;
        };
        let include = include.trim_start();
        // SDK headers are covered by the compiler and SDK identities below.
        if include.starts_with('<') {
            continue;
        }
        let Some((name, _)) = include.strip_prefix('"').and_then(|s| s.split_once('"')) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported Metal include in {}: {line}", path.display()),
            ));
        };
        if name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty Metal include",
            ));
        }
        hash_key_part(hasher, "include", name.as_bytes());
        hash_source_tree(
            &path.parent().unwrap().join(name),
            "dependency",
            hasher,
            seen,
            dependencies,
        )?;
    }
    Ok(())
}

/// Hash the complete local include tree, not only the top-level Metal source.
/// Include order is deterministic; canonical paths prevent cycles but do not
/// enter the digest, so identical worktrees retain content-based AIR reuse.
pub(crate) fn source_cache_key(
    source: &Path,
    language: &str,
    target: &str,
    compiler_identity: &str,
    sdk_version: &str,
) -> io::Result<SourceCacheKey> {
    let mut hasher = Sha256::new();
    hash_key_part(&mut hasher, "language", language.as_bytes());
    hash_key_part(&mut hasher, "target", target.as_bytes());
    hash_key_part(&mut hasher, "compiler", compiler_identity.as_bytes());
    hash_key_part(&mut hasher, "sdk", sdk_version.as_bytes());
    hash_key_part(&mut hasher, "flags", b"-O3 -ffast-math");
    let mut dependencies = Vec::new();
    hash_source_tree(
        source,
        "source",
        &mut hasher,
        &mut BTreeSet::new(),
        &mut dependencies,
    )?;
    Ok(SourceCacheKey {
        hash: hex_digest(&hasher.finalize()),
        dependencies,
    })
}
