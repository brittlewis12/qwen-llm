//! Stamps build identity into `qwen-bench` JSON output.
//!
//! Normal builds derive identity from Git. Reproducible/distribution builds may
//! set `QWEN_BUILD_COMMIT` (a full 40- or 64-hex object id),
//! `QWEN_BUILD_DIRTY` (`0` or `1`), and `QWEN_BUILD_SOURCE_STATE` (the exact
//! source-state digest). Environment assertions are independently verified
//! against Git when available. Invalid, incomplete, or unverifiable overrides are stamped as
//! unverifiable so the benchmark CLI can fail closed before loading a model.

mod source_identity;

use source_identity::{
    full_object_id, git_dirty, git_text, tracked_source_state, valid_source_state,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn git_path(manifest_dir: &Path, name: &str) -> Option<PathBuf> {
    let value = git_text(manifest_dir, &["rev-parse", "--git-path", name])?;
    let path = PathBuf::from(value);
    Some(if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    })
}

fn watch_git_metadata(manifest_dir: &Path) {
    for name in ["HEAD", "index", "packed-refs", "refs", "reftable"] {
        if let Some(path) = git_path(manifest_dir, name)
            && path.exists()
        {
            // Cargo treats a missing rerun path as changed on every invocation.
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    if let Some(symbolic_ref) = git_text(manifest_dir, &["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git_path(manifest_dir, &symbolic_ref)
        && path.exists()
    {
        // Watching the symbolic ref target is the critical part: committing on
        // a branch changes refs/heads/<branch>, while .git/HEAD stays unchanged.
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn watch_source_files(manifest_dir: &Path) {
    let Some(root) = git_text(manifest_dir, &["rev-parse", "--show-toplevel"]) else {
        return;
    };
    let root = Path::new(&root);
    for args in [
        &["ls-files", "-z"][..],
        &["ls-files", "--others", "--exclude-standard", "-z"][..],
    ] {
        let Ok(output) = Command::new("git").args(args).current_dir(root).output() else {
            return;
        };
        if !output.status.success() {
            return;
        }
        for path in output.stdout.split(|byte| *byte == 0) {
            if path.is_empty() {
                continue;
            }
            #[cfg(unix)]
            let path = {
                use std::ffi::OsStr;
                use std::os::unix::ffi::OsStrExt;
                root.join(OsStr::from_bytes(path))
            };
            #[cfg(not(unix))]
            let path = match std::str::from_utf8(path) {
                Ok(path) => root.join(path),
                Err(_) => continue,
            };
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=QWEN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=QWEN_BUILD_DIRTY");
    println!("cargo:rerun-if-env-changed=QWEN_BUILD_SOURCE_STATE");
    watch_git_metadata(&manifest_dir);
    watch_source_files(&manifest_dir);

    let commit_override = std::env::var("QWEN_BUILD_COMMIT").ok();
    let dirty_override = std::env::var("QWEN_BUILD_DIRTY").ok();
    let state_override = std::env::var("QWEN_BUILD_SOURCE_STATE").ok();
    let git_identity = match (
        git_text(&manifest_dir, &["rev-parse", "HEAD"]),
        git_dirty(&manifest_dir),
        tracked_source_state(&manifest_dir),
    ) {
        (Some(commit), Some(dirty), Some(state))
            if full_object_id(&commit) && valid_source_state(&state) =>
        {
            Some((commit.to_ascii_lowercase(), dirty, state))
        }
        _ => None,
    };
    let (commit, dirty, state, source, error) =
        match (commit_override, dirty_override, state_override) {
            (None, None, None) => match git_identity {
                Some((commit, dirty, state)) => (commit, Some(dirty), state, "git", "none"),
                _ => (
                    "unknown".to_string(),
                    None,
                    "unknown".to_string(),
                    "unknown",
                    "git_identity_unavailable",
                ),
            },
            (Some(commit), Some(dirty), Some(state))
                if full_object_id(&commit)
                    && matches!(dirty.as_str(), "0" | "1")
                    && valid_source_state(&state) =>
            {
                let asserted = (
                    commit.to_ascii_lowercase(),
                    dirty == "1",
                    state.to_ascii_lowercase(),
                );
                let (source, error) = match git_identity {
                    Some(actual) if actual == asserted => ("environment-verified", "none"),
                    Some(_) => ("environment-mismatch", "environment_identity_mismatch"),
                    None => ("environment-unverified", "environment_identity_unverified"),
                };
                (asserted.0, Some(asserted.1), asserted.2, source, error)
            }
            (Some(_), Some(_), Some(_)) => (
                "unknown".to_string(),
                None,
                "unknown".to_string(),
                "environment-invalid",
                "invalid_identity_override",
            ),
            _ => (
                "unknown".to_string(),
                None,
                "unknown".to_string(),
                "environment-invalid",
                "identity_override_triple_required",
            ),
        };

    let short = if full_object_id(&commit) {
        &commit[..9]
    } else {
        "unknown"
    };
    let dirty = match dirty {
        Some(true) => "1",
        Some(false) => "0",
        None => "unknown",
    };
    println!("cargo:rustc-env=QWEN_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=QWEN_BUILD_COMMIT_SHORT={short}");
    println!("cargo:rustc-env=QWEN_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=QWEN_BUILD_SOURCE_STATE={state}");
    println!("cargo:rustc-env=QWEN_BUILD_STAMP_SOURCE={source}");
    println!("cargo:rustc-env=QWEN_BUILD_STAMP_ERROR={error}");
}
