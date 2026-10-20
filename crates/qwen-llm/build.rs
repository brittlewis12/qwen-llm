//! Build script that compiles all `.metal` kernel sources into a single
//! `.metallib` archive and embeds the bytes into the library via
//! `OUT_DIR/kernels.metallib`.
//!
//! Invoked at `cargo build` time. Recompiles when inputs or the worktree change.

use std::path::{Path, PathBuf};
use std::process::Command;

// The oldest language every product and research kernel compiles under, so
// the libraries load on macOS 14. Raise it only for a kernel that needs a
// newer language feature.
const DEFAULT_METALLIB_TARGET: &str = "14.0";
const METAL_LANGUAGE: &str = "metal3.1";
const TENSOR_METAL_LANGUAGE: &str = "metal4.0";
const TENSOR_METAL_TARGET: &str = "26.0";

fn main() -> anyhow::Result<()> {
    // MTLResidencySetDescriptor is available on macOS 15 and newer. Keep the
    // framework reference weak so release binaries start on the macOS 14 floor.
    println!("cargo:rustc-link-arg=-Wl,-weak_framework,Metal");
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("could not resolve workspace root from {manifest_dir:?}"))?
        .to_path_buf();
    let kernels_dir = workspace_root.join("kernels");
    let watched_kernels_dir = PathBuf::from("../../kernels");

    // Keep kernel watches relative, but bind cached output to this worktree.
    // Cargo strips the current package root from rerun paths when fingerprinting:
    // this absolute path becomes `build.rs` here, but stays absolute when another
    // worktree reads the cached output, forcing a rerun even with older mtimes.
    // Keep the manifest lookup at runtime: Cargo filters CARGO_MANIFEST_DIR out
    // of env! dep-info, so a reused build-script executable must use its new root.
    println!("cargo:rerun-if-changed={}", watched_kernels_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("build.rs").display()
    );
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    // Record the exact toolchain and SDK selected for the same macOS SDK
    // invocation used below to compile every metallib.
    let compiler_identity = xcrun_output(&["-sdk", "macosx", "metal", "--version"])?
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    anyhow::ensure!(
        !compiler_identity.is_empty(),
        "xcrun metal --version returned no compiler identity"
    );
    let sdk_version = xcrun_output(&["-sdk", "macosx", "--show-sdk-version"])?
        .trim()
        .to_string();
    anyhow::ensure!(
        !sdk_version.is_empty(),
        "xcrun returned an empty macOS SDK version"
    );
    println!("cargo:rustc-env=QWEN_METALLIB_COMPILER_IDENTITY={compiler_identity}");
    println!("cargo:rustc-env=QWEN_METALLIB_SDK_VERSION={sdk_version}");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    // Select Metal 4 tiers from compiler support, not SDK branding. Each
    // probe compiles real MSL with the standard the corresponding library
    // will use; a failed optional tier is recorded and reported, not fatal.
    let tensor_dir = workspace_root.join("kernels/tensor");
    let watched_tensor_dir = watched_kernels_dir.join("tensor");
    let tensor_out = out_dir.join("kernels_tensor.metallib");
    let tensor4_probe = probe_msl("metal4.0", "26.0", &out_dir, "tensor4");
    let tensor41_probe = probe_msl("metal4.1", "27.0", &out_dir, "tensor41");
    let tensor_tier = if tensor4_probe.is_ok() {
        "metal4.0@26.0"
    } else {
        "none"
    };
    let tensor40_probe_status = if tensor4_probe.is_ok() {
        "accepted"
    } else {
        "rejected"
    };
    let tensor41_probe_status = if tensor41_probe.is_ok() {
        "accepted"
    } else {
        "rejected"
    };
    println!("cargo:rustc-env=QWEN_TENSOR_METALLIB_TIERS={tensor_tier}");
    println!("cargo:rustc-env=QWEN_TENSOR_METAL40_PROBE={tensor40_probe_status}");
    println!("cargo:rustc-env=QWEN_TENSOR_METAL41_PROBE={tensor41_probe_status}");
    println!(
        "cargo:rustc-env=QWEN_TENSOR_METALLIB_TARGET={}",
        if tensor4_probe.is_ok() {
            TENSOR_METAL_TARGET
        } else {
            ""
        }
    );
    match tensor4_probe {
        Ok(()) => compile_metallib_with_language(
            &tensor_dir,
            &watched_tensor_dir,
            &out_dir,
            "kernels_tensor",
            TENSOR_METAL_TARGET,
            TENSOR_METAL_LANGUAGE,
        )?,
        Err(reason) => {
            println!(
                "cargo:warning=Metal tensor-op kernels skipped: MSL 4.0 compiler probe failed: {reason}"
            );
            std::fs::write(tensor_out, [])?;
        }
    }
    if let Err(reason) = tensor41_probe {
        println!("cargo:warning=Metal 4.1 tensor-op tier skipped: compiler probe failed: {reason}");
    } else {
        println!(
            "cargo:warning=Metal 4.1 compiler probe passed; no 4.1-only tensor sources are present"
        );
    }

    // Pin the language and deployment target: without them the compiler
    // derives the target from the build host, so a library built on a newer
    // macOS may not load on an older one. MSL 4 raises the target to macOS 26
    // even with -mmacosx-version-min, so MSL 4 kernels need their own library
    // built for macOS 26 rather than living in this one.
    let target = match std::env::var("MACOSX_DEPLOYMENT_TARGET") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => DEFAULT_METALLIB_TARGET.to_string(),
        Err(std::env::VarError::NotUnicode(value)) => {
            anyhow::bail!("MACOSX_DEPLOYMENT_TARGET is not valid UTF-8: {value:?}")
        }
    };
    let target_suffix = normalize_target_version(&target)?;
    // The AIR version in the triple belongs to the compiler; only the macOS
    // part is checked.
    let metal_target = format!("-apple-macosx{target_suffix}");
    println!("cargo:rustc-env=QWEN_PRODUCT_METALLIB_TARGET={target}");
    println!("cargo:rustc-env=QWEN_RESEARCH_METALLIB_TARGET={target}");

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

fn xcrun_output(args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new("xcrun").args(args).output()?;
    anyhow::ensure!(
        output.status.success(),
        "xcrun {args:?} failed with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn compile_metallib(
    dir: &Path,
    watched_dir: &Path,
    out_dir: &Path,
    name: &str,
    target: &str,
) -> anyhow::Result<()> {
    compile_metallib_with_language(dir, watched_dir, out_dir, name, target, METAL_LANGUAGE)
}

fn compile_metallib_with_language(
    dir: &Path,
    watched_dir: &Path,
    out_dir: &Path,
    name: &str,
    target: &str,
    language: &str,
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
            .arg(format!("-std={language}"))
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

fn probe_msl(language: &str, target: &str, out_dir: &Path, name: &str) -> Result<(), String> {
    let source = out_dir.join(format!("{name}_probe.metal"));
    let air = out_dir.join(format!("{name}_probe.air"));
    std::fs::write(
        &source,
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void qwen_tensor_probe() {}\n",
    )
    .map_err(|error| error.to_string())?;
    let output = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            &format!("-std={language}"),
            &format!("-mmacosx-version-min={target}"),
        ])
        .arg(&source)
        .arg("-o")
        .arg(&air)
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
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
    // The driver can report an error (e.g. an out-of-range minor version)
    // and still exit 0 with a matching triple.
    if transcript.contains("error:") {
        anyhow::bail!(
            "Metal target probe reported an error for requested macOS {target}:\n{transcript}"
        );
    }
    let tokens: Vec<_> = transcript
        .split_whitespace()
        .map(|token| token.trim_matches('"'))
        .collect();
    let triples: Vec<_> = tokens
        .windows(2)
        .filter_map(|pair| (pair[0] == "-triple").then_some(pair[1]))
        .collect();
    let [triple] = triples.as_slice() else {
        anyhow::bail!(
            "Metal target probe: expected one -triple in the compiler transcript, found {triples:?}:\n{transcript}"
        );
    };
    if !(triple.starts_with("air64") && triple.ends_with(expected)) {
        anyhow::bail!(
            "Metal compiler effective target mismatch: requested macOS {target} (triple ending {expected}), observed {triple}"
        );
    }
    Ok(())
}

/// `15`, `15.0` and `015.0.0` all normalize to the compiler's `15.0.0`.
fn normalize_target_version(target: &str) -> anyhow::Result<String> {
    let parts = target
        .split('.')
        .map(|part| {
            // Digits only: u32 parsing alone would accept "+15".
            (!part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| part.parse::<u32>().ok())
                .flatten()
        })
        .collect::<Option<Vec<_>>>()
        .filter(|parts| (1..=3).contains(&parts.len()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "invalid MACOSX_DEPLOYMENT_TARGET {target:?}; expected a numeric macOS version such as 15.0"
            )
        })?;
    let part = |index: usize| parts.get(index).copied().unwrap_or(0);
    Ok(format!("{}.{}.{}", part(0), part(1), part(2)))
}
