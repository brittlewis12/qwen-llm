//! Build script that compiles all `.metal` kernel sources into a single
//! `.metallib` archive and embeds the bytes into the library via
//! `OUT_DIR/kernels.metallib`.
//!
//! Invoked at `cargo build` time. Recompiles only when `kernels/` changes.

use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_METALLIB_TARGET: &str = "15.0";
const METAL_LANGUAGE: &str = "metal3.2";

fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("could not resolve workspace root from {manifest_dir:?}"))?
        .to_path_buf();
    let kernels_dir = workspace_root.join("kernels");
    let watched_kernels_dir = PathBuf::from("../../kernels");

    // Keep the fingerprint portable when worktrees share CARGO_TARGET_DIR.
    println!("cargo:rerun-if-changed={}", watched_kernels_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    // Pin the language and deployment target: without them the compiler
    // derives the target from the build host, so a library built on a newer
    // macOS may not load on macOS 15. MSL 4 raises the target to macOS 26
    // even with -mmacosx-version-min, so MSL 4 kernels need their own library
    // built for macOS 26 rather than living in this one.
    let target = std::env::var("MACOSX_DEPLOYMENT_TARGET")
        .unwrap_or_else(|_| DEFAULT_METALLIB_TARGET.to_string());
    let target_suffix = normalize_target_version(&target)?;
    // The AIR version in the triple belongs to the compiler; only the macOS
    // part is checked.
    let metal_target = format!("-apple-macosx{target_suffix}");
    println!("cargo:rustc-env=QWEN_PRODUCT_METALLIB_TARGET={target}");
    println!("cargo:rustc-env=QWEN_RESEARCH_METALLIB_TARGET={target}");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    verify_effective_target(&kernels_dir, &target, &metal_target)?;
    // Product kernels live at the top level; bench/research-only probes under
    // kernels/research/ compile into a second metallib the runtime loads
    // lazily, so the product library stays an honest inventory.
    compile_metallib(
        &kernels_dir,
        &watched_kernels_dir,
        &out_dir,
        "kernels",
        &target,
    )?;
    compile_metallib(
        &kernels_dir.join("research"),
        &watched_kernels_dir.join("research"),
        &out_dir,
        "kernels_research",
        &target,
    )?;
    Ok(())
}

fn compile_metallib(
    dir: &Path,
    watched_dir: &Path,
    out_dir: &Path,
    name: &str,
    target: &str,
) -> anyhow::Result<()> {
    let metallib = out_dir.join(format!("{name}.metallib"));
    if !dir.exists() {
        std::fs::write(&metallib, [])?;
        return Ok(());
    }
    println!("cargo:rerun-if-changed={}", watched_dir.display());
    let mut metal_files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    metal_files.sort();
    for path in &metal_files {
        println!(
            "cargo:rerun-if-changed={}",
            watched_dir.join(path.file_name().unwrap()).display()
        );
    }
    if metal_files.is_empty() {
        std::fs::write(&metallib, [])?;
        return Ok(());
    }
    let air_files: Vec<PathBuf> = metal_files
        .iter()
        .map(|src| {
            let stem = src.file_stem().unwrap().to_string_lossy();
            out_dir.join(format!("{name}_{stem}.air"))
        })
        .collect();
    for (src, air) in metal_files.iter().zip(air_files.iter()) {
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg("-O3")
            .arg("-ffast-math")
            .arg(format!("-std={METAL_LANGUAGE}"))
            .arg(format!("-mmacosx-version-min={target}"))
            .arg(src)
            .arg("-o")
            .arg(air)
            .status()?;
        if !status.success() {
            anyhow::bail!("metal compile failed for {}", src.display());
        }
    }
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(&metallib)
        .status()?;
    if !status.success() {
        anyhow::bail!("metallib link failed for {name}");
    }
    Ok(())
}

fn verify_effective_target(kernels_dir: &Path, target: &str, expected: &str) -> anyhow::Result<()> {
    let source = std::fs::read_dir(kernels_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("metal"))
        .ok_or_else(|| {
            anyhow::anyhow!("cannot verify Metal target: no product .metal source found")
        })?;
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-###", "-c"])
        .arg("-O3")
        .arg("-ffast-math")
        .arg(format!("-std={METAL_LANGUAGE}"))
        .arg(format!("-mmacosx-version-min={target}"))
        .arg(&source)
        .arg("-o")
        .arg("/dev/null")
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "Metal target probe failed for requested macOS {target}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let transcript = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let tokens: Vec<_> = transcript
        .split_whitespace()
        .map(|token| token.trim_matches('"'))
        .collect();
    let observed = tokens
        .windows(2)
        .find_map(|pair| (pair[0] == "-triple").then_some(pair[1]));
    if !observed.is_some_and(|triple| triple.starts_with("air64") && triple.ends_with(expected)) {
        anyhow::bail!(
            "Metal compiler effective target mismatch: requested macOS {target} (triple ending {expected}), observed {:?}",
            observed
        );
    }
    Ok(())
}

fn normalize_target_version(target: &str) -> anyhow::Result<String> {
    let components: Vec<_> = target.split('.').collect();
    if components.is_empty()
        || components.len() > 3
        || components.iter().any(|part| part.parse::<u32>().is_err())
    {
        anyhow::bail!(
            "invalid MACOSX_DEPLOYMENT_TARGET {target:?}; expected a numeric macOS version such as 15.0"
        );
    }
    Ok(match components.len() {
        1 => format!("{}.0.0", components[0]),
        2 => format!("{}.{}.0", components[0], components[1]),
        _ => target.to_string(),
    })
}
