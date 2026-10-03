//! Startup-validated, CPU-resident prebuilt web assets. Never invokes Bun.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path};

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_ASSETS_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ASSETS: usize = 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    entry: String,
    files: Vec<ManifestFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ManifestFile {
    path: String,
    content_type: String,
    bytes: u64,
}

struct Asset {
    content_type: String,
    bytes: Vec<u8>,
}

pub(crate) struct WebAssets {
    entry: String,
    files: BTreeMap<String, Asset>,
}

fn valid_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains(['\\', '%', '?', '#', '\0'])
        && !path.starts_with('/')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn read_asset(root: &Path, path: &str, bound: u64) -> Result<Vec<u8>> {
    ensure!(valid_path(path), "invalid web asset path");
    let mut current = root.to_path_buf();
    for part in Path::new(path).components() {
        current.push(part);
        ensure!(
            !std::fs::symlink_metadata(&current)?
                .file_type()
                .is_symlink(),
            "web asset symlink rejected"
        );
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(&current)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= bound,
        "invalid web asset size or file type"
    );
    let mut bytes = Vec::new();
    file.take(bound + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= bound, "web asset grew beyond limit");
    Ok(bytes)
}

impl WebAssets {
    pub(crate) fn open(root: &Path) -> Result<Self> {
        let root = root.canonicalize().context("resolve --web-root")?;
        ensure!(root.is_dir(), "--web-root must be a directory");
        let manifest: Manifest = serde_json::from_slice(&read_asset(
            &root,
            "asset-manifest.json",
            MAX_MANIFEST_BYTES,
        )?)
        .context("parse web asset-manifest.json")?;
        ensure!(
            manifest.version == 1,
            "unsupported web asset manifest version"
        );
        ensure!(
            manifest.entry == "index.html",
            "web manifest entry must be index.html"
        );
        ensure!(
            !manifest.files.is_empty() && manifest.files.len() <= MAX_ASSETS,
            "invalid web asset count"
        );
        let mut files = BTreeMap::new();
        let mut total = 0u64;
        for file in manifest.files {
            total = total
                .checked_add(file.bytes)
                .context("web asset size overflow")?;
            ensure!(
                total <= MAX_ASSETS_BYTES,
                "web assets exceed CPU memory budget"
            );
            ensure!(
                !file.content_type.is_empty()
                    && file.content_type.bytes().all(|b| (32..127).contains(&b)),
                "invalid web content type"
            );
            ensure!(
                file.path != "asset-manifest.json"
                    && file.path != "v1"
                    && !file.path.starts_with("v1/"),
                "reserved web asset path"
            );
            let bytes = read_asset(&root, &file.path, file.bytes)?;
            ensure!(
                bytes.len() as u64 == file.bytes,
                "web asset size differs from manifest"
            );
            ensure!(
                files
                    .insert(
                        file.path,
                        Asset {
                            content_type: file.content_type,
                            bytes
                        }
                    )
                    .is_none(),
                "duplicate web asset path"
            );
        }
        ensure!(
            files.contains_key(&manifest.entry),
            "web manifest entry is missing"
        );
        Ok(Self {
            entry: manifest.entry,
            files,
        })
    }

    fn asset(&self, method: &str, path: &str) -> Option<&Asset> {
        if !matches!(method, "GET" | "HEAD") {
            return None;
        }
        let path = path.split_once('?').map_or(path, |(path, _)| path);
        if path == "/v1" || path.starts_with("/v1/") {
            return None;
        }
        let key = if path == "/" {
            self.entry.as_str()
        } else {
            path.strip_prefix('/')?
        };
        self.files.get(key)
    }

    pub(super) fn matches(&self, method: &str, path: &str) -> bool {
        self.asset(method, path).is_some()
    }

    pub(crate) fn serve(
        &self,
        method: &str,
        path: &str,
        mut stream: &TcpStream,
    ) -> std::io::Result<bool> {
        let Some(asset) = self.asset(method, path) else {
            return Ok(false);
        };
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: {}\r\ncontent-length: {}\r\ncache-control: no-cache\r\nx-content-type-options: nosniff\r\nconnection: close\r\n\r\n",
            asset.content_type,
            asset.bytes.len()
        )?;
        if method == "GET" {
            stream.write_all(&asset.bytes)?;
        }
        stream.flush()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot(std::path::PathBuf);
    impl TestRoot {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "qwen-web-assets-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn manifest(&self, bytes: usize) {
            std::fs::write(self.0.join("asset-manifest.json"), serde_json::to_vec(&serde_json::json!({
                "version": 1, "entry": "index.html", "files": [{"path":"index.html","contentType":"text/html;charset=utf-8","bytes":bytes}]
            })).unwrap()).unwrap();
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn loads_exact_manifest_bytes_and_rejects_size_drift() {
        let root = TestRoot::new();
        root.manifest(5);
        std::fs::write(root.0.join("index.html"), b"hello").unwrap();
        let assets = WebAssets::open(&root.0).unwrap();
        assert_eq!(assets.files["index.html"].bytes, b"hello");
        std::fs::write(root.0.join("index.html"), b"changed").unwrap();
        assert!(WebAssets::open(&root.0).is_err());
        assert_eq!(assets.files["index.html"].bytes, b"hello");
    }

    #[test]
    fn rejects_manifest_asset_symlinks() {
        let root = TestRoot::new();
        root.manifest(5);
        std::fs::write(root.0.join("other.html"), b"hello").unwrap();
        std::os::unix::fs::symlink("other.html", root.0.join("index.html")).unwrap();
        assert!(WebAssets::open(&root.0).is_err());
    }

    #[test]
    fn api_namespace_cannot_be_shadowed_by_static_files() {
        let root = TestRoot::new();
        for path in ["v1", "v1/unknown", "v1/lens/capabilities"] {
            std::fs::write(root.0.join("asset-manifest.json"), serde_json::to_vec(&serde_json::json!({
                "version":1,"entry":"index.html","files":[{"path":path,"contentType":"text/html","bytes":0}]
            })).unwrap()).unwrap();
            assert!(
                WebAssets::open(&root.0)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("reserved web asset path")
            );
        }
        root.manifest(0);
        std::fs::write(root.0.join("index.html"), b"").unwrap();
        let assets = WebAssets::open(&root.0).unwrap();
        for method in ["GET", "HEAD", "POST", "DELETE"] {
            for path in ["/v1", "/v1/unknown", "/v1?ignored=1"] {
                assert!(!assets.matches(method, path));
            }
        }
    }

    #[test]
    fn rejects_ambiguous_asset_paths() {
        for path in [
            "",
            "/index.html",
            "../secret",
            "a/../b",
            "a//b",
            "a/./b",
            "a\\b",
            "%2e%2e/x",
            "x?y",
            "x#y",
        ] {
            assert!(!valid_path(path), "{path:?}");
        }
        assert!(valid_path("assets/main-123.js"));
    }
}
