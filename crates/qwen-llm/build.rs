//! Build script that compiles all `.metal` kernel sources into a single
//! `.metallib` archive and embeds the bytes into the library via
//! `OUT_DIR/kernels.metallib`.
//!
//! Invoked at `cargo build` time. Recompiles only when `kernels/` changes.

use std::path::PathBuf;
use std::process::Command;

fn main() -> anyhow::Result<()> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("could not resolve workspace root from {manifest_dir:?}"))?
        .to_path_buf();
    let kernels_dir = workspace_root.join("kernels");

    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-changed=build.rs");

    if !kernels_dir.exists() {
        // No kernels yet; emit an empty stub so lib.rs can include_bytes!().
        let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
        std::fs::write(out_dir.join("kernels.metallib"), [])?;
        return Ok(());
    }

    let mut metal_files: Vec<PathBuf> = std::fs::read_dir(&kernels_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    metal_files.sort();
    for path in &metal_files {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    if metal_files.is_empty() {
        std::fs::write(out_dir.join("kernels.metallib"), [])?;
        return Ok(());
    }

    let air_files: Vec<PathBuf> = metal_files
        .iter()
        .map(|src| {
            let stem = src.file_stem().unwrap().to_string_lossy();
            out_dir.join(format!("{stem}.air"))
        })
        .collect();

    for (src, air) in metal_files.iter().zip(air_files.iter()) {
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg("-O3")
            .arg("-ffast-math")
            .arg(src)
            .arg("-o")
            .arg(air)
            .status()?;
        if !status.success() {
            anyhow::bail!("metal compile failed for {}", src.display());
        }
    }

    let metallib = out_dir.join("kernels.metallib");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(&metallib)
        .status()?;
    if !status.success() {
        anyhow::bail!("metallib link failed");
    }

    Ok(())
}
