//! Stamps `QWEN_BUILD_COMMIT` and `QWEN_BUILD_DIRTY` into the binary so
//! `qwen-bench -o json` can emit a build identity matching lcpp's
//! `build_commit` field. Commit resolution: env var, then `git rev-parse`,
//! then `"unknown"`. Dirty defaults to checking `git status --porcelain`.

use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    let git_dir_candidates = [
        format!("{manifest_dir}/../../.git/HEAD"),
        format!("{manifest_dir}/../../.git/index"),
    ];
    for p in &git_dir_candidates {
        if std::path::Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    println!("cargo:rerun-if-env-changed=QWEN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=QWEN_BUILD_DIRTY");

    let commit = std::env::var("QWEN_BUILD_COMMIT")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short=9", "HEAD"])
                .current_dir(&manifest_dir)
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                    } else {
                        None
                    }
                })
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=QWEN_BUILD_COMMIT={commit}");

    let dirty = std::env::var("QWEN_BUILD_DIRTY")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or_else(|| {
            Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(&manifest_dir)
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false)
        });
    println!("cargo:rustc-env=QWEN_BUILD_DIRTY={}", dirty as u8);
}
