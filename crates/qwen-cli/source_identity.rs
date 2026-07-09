use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const SOURCE_STATE_PREFIX: &str = "git-source-sha256-v2:";

fn git_bytes(repo: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

pub fn git_text(repo: &Path, args: &[&str]) -> Option<String> {
    let output = git_bytes(repo, args)?;
    let value = String::from_utf8(output).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn git_root(repo: &Path) -> Option<PathBuf> {
    git_text(repo, &["rev-parse", "--show-toplevel"]).map(PathBuf::from)
}

fn index_has_hidden_worktree_entries(repo: &Path) -> Option<bool> {
    let entries = git_bytes(repo, &["ls-files", "-v", "-z"])?;
    Some(entries.split(|byte| *byte == 0).any(|entry| {
        entry
            .first()
            .is_some_and(|tag| tag.is_ascii_lowercase() || *tag == b'S')
    }))
}

pub fn git_dirty(repo: &Path) -> Option<bool> {
    let root = git_root(repo)?;
    let status = git_bytes(&root, &["status", "--porcelain", "--untracked-files=all"])?;
    Some(!status.is_empty() || index_has_hidden_worktree_entries(&root)?)
}

pub fn full_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn valid_source_state(value: &str) -> bool {
    value
        .strip_prefix(SOURCE_STATE_PREFIX)
        .is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
}

fn hash_section(hasher: &mut Sha256, label: &[u8], value: &[u8]) {
    hasher.update((label.len() as u64).to_be_bytes());
    hasher.update(label);
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

#[cfg(unix)]
fn path_from_git_bytes(root: &Path, value: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    root.join(OsStr::from_bytes(value))
}

#[cfg(not(unix))]
fn path_from_git_bytes(root: &Path, value: &[u8]) -> Option<PathBuf> {
    std::str::from_utf8(value)
        .ok()
        .map(|value| root.join(value))
}

#[cfg(unix)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(not(unix))]
fn os_str_bytes(value: &OsStr) -> Option<Vec<u8>> {
    value.to_str().map(|value| value.as_bytes().to_vec())
}

fn hash_file(path: &Path) -> Option<(u64, [u8; 32])> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        size = size.checked_add(read as u64)?;
        hasher.update(&buffer[..read]);
    }
    Some((size, hasher.finalize().into()))
}

fn hash_worktree_entry(
    hasher: &mut Sha256,
    root: &Path,
    scope: &[u8],
    relative: &[u8],
) -> Option<()> {
    hash_section(hasher, b"entry-scope", scope);
    hash_section(hasher, b"entry-path", relative);
    #[cfg(unix)]
    let path = path_from_git_bytes(root, relative);
    #[cfg(not(unix))]
    let path = path_from_git_bytes(root, relative)?;

    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            hash_section(hasher, b"entry-kind", b"missing");
            return Some(());
        }
        Err(_) => return None,
    };
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        hash_section(hasher, b"entry-kind", b"symlink");
        let target = fs::read_link(path).ok()?;
        #[cfg(unix)]
        let target = os_str_bytes(target.as_os_str());
        #[cfg(not(unix))]
        let target = os_str_bytes(target.as_os_str())?;
        hash_section(hasher, b"entry-content", &target);
    } else if file_type.is_file() {
        hash_section(hasher, b"entry-kind", b"file");
        #[cfg(unix)]
        let executable = {
            use std::os::unix::fs::PermissionsExt;
            u8::from(metadata.permissions().mode() & 0o111 != 0)
        };
        #[cfg(not(unix))]
        let executable = 0u8;
        hash_section(hasher, b"entry-executable", &[executable]);
        let (size, content_hash) = hash_file(&path)?;
        hash_section(hasher, b"entry-size", &size.to_be_bytes());
        hash_section(hasher, b"entry-content-sha256", &content_hash);
    } else {
        return None;
    }
    Some(())
}

pub fn tracked_source_state(repo: &Path) -> Option<String> {
    let root = git_root(repo)?;
    let head = git_bytes(&root, &["rev-parse", "HEAD"])?;
    let index = git_bytes(&root, &["ls-files", "--stage", "-z"])?;
    let index_flags = git_bytes(&root, &["ls-files", "-v", "-z"])?;
    let tracked = git_bytes(&root, &["ls-files", "-z"])?;
    let untracked = git_bytes(&root, &["ls-files", "--others", "--exclude-standard", "-z"])?;

    let mut hasher = Sha256::new();
    hasher.update(b"qwen-git-source-state-v2\0");
    hash_section(&mut hasher, b"head", &head);
    hash_section(&mut hasher, b"index", &index);
    hash_section(&mut hasher, b"index-flags", &index_flags);
    hash_section(&mut hasher, b"tracked-paths", &tracked);
    hash_section(&mut hasher, b"untracked-paths", &untracked);
    for path in tracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        hash_worktree_entry(&mut hasher, &root, b"tracked", path)?;
    }
    for path in untracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        hash_worktree_entry(&mut hasher, &root, b"untracked", path)?;
    }
    Some(format!("{SOURCE_STATE_PREFIX}{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_checkout_has_versioned_source_state() {
        let state =
            tracked_source_state(Path::new(env!("CARGO_MANIFEST_DIR"))).expect("source state");
        assert!(valid_source_state(&state));
    }
}
