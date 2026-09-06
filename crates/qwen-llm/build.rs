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
    let watched_kernels_dir = PathBuf::from("../../kernels");

    // Keep the fingerprint portable when worktrees share CARGO_TARGET_DIR.
    println!("cargo:rerun-if-changed={}", watched_kernels_dir.display());
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    // Product kernels live at the top level; bench/research-only probes under
    // kernels/research/ compile into a second metallib the runtime loads
    // lazily, so the product library stays an honest inventory.
    compile_metallib(&kernels_dir, &watched_kernels_dir, &out_dir, "kernels")?;
    compile_metallib(
        &kernels_dir.join("research"),
        &watched_kernels_dir.join("research"),
        &out_dir,
        "kernels_research",
    )?;
    Ok(())
}

fn compile_metallib(
    dir: &PathBuf,
    watched_dir: &PathBuf,
    out_dir: &PathBuf,
    name: &str,
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
