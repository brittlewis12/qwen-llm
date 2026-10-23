use super::*;
use std::process::Command;

pub(super) fn environment() -> AResult<()> {
    ensure!(
        !cfg!(debug_assertions) && cfg!(target_endian = "little"),
        "release little-endian test binary required"
    );
    for name in ["MTL_DEBUG_LAYER", "QWEN_METAL_LEASE_WAIT"] {
        ensure!(
            std::env::var(name).as_deref() == Ok("1"),
            "{name}=1 required"
        );
    }
    for (name, value) in std::env::vars_os() {
        let name = name.to_string_lossy();
        let expected = match name.as_ref() {
            "QWEN_PRODUCT_METALLIB_TARGET" => Some(crate::PRODUCT_METALLIB_TARGET),
            "QWEN_RESEARCH_METALLIB_TARGET" => Some(crate::RESEARCH_METALLIB_TARGET),
            "QWEN_TENSOR_METALLIB_TARGET" => Some(crate::TENSOR_METALLIB_TARGET),
            "QWEN_TENSOR_METALLIB_TIERS" => Some(crate::TENSOR_METALLIB_TIERS),
            "QWEN_TENSOR_METAL40_PROBE" => Some(crate::TENSOR_METAL40_PROBE),
            "QWEN_TENSOR_METAL41_PROBE" => Some(crate::TENSOR_METAL41_PROBE),
            "QWEN_METALLIB_COMPILER_IDENTITY" => Some(crate::metallib_build_identity().0),
            "QWEN_METALLIB_SDK_VERSION" => Some(crate::metallib_build_identity().1),
            _ => None,
        };
        if let Some(expected) = expected {
            ensure!(
                value == std::ffi::OsStr::new(expected),
                "build provenance mismatch: {name}"
            );
        } else {
            ensure!(
                !name.starts_with("QWEN_") || name == "QWEN_METAL_LEASE_WAIT",
                "remove runtime/diagnostic override {name}"
            );
        }
    }
    ensure!(
        !prefill::F32_LINEAR_ROWS.with(|f| f.get())
            && !prefill::ROWWISE_LINEAR_ROWS.with(|f| f.get()),
        "linear override already active"
    );
    ensure!(
        !metal::dispatch_census_is_active(),
        "dispatch observer already active"
    );
    Ok(())
}

pub(super) fn profile(g: &crate::gguf::GgufFile, p: &v41_gguf::Plan) -> AResult<()> {
    ensure!(
        g.architecture().as_deref() == Some("deepseek41")
            && format!("{:?}", p.config) == format!("{:?}", Config::default())
            && p.schedule == v41_gguf::Schedule::default()
            && p.engram_hash == engram::HashConfig::default()
            && p.raw_pad == 2,
        "full stock 1017 geometry/schedule/hash/pad required; not REAP-272"
    );
    p.schedule.validate()?;
    p.engram_hash
        .validate()
        .map_err(|e| anyhow::anyhow!("hash profile: {e:?}"))?;
    let banks: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| t.name.ends_with("_exps.weight"))
        .collect();
    // Stock UD-IQ3_XXS itself includes residual IQ2 banks. Its label is
    // only a profile guard; exact identity is the frozen layout + shard stamps.
    ensure!(
        g.shards
            .first()
            .and_then(|s| s.path.file_name())
            .is_some_and(|n| n.to_string_lossy().to_ascii_uppercase().contains("IQ3"))
            && banks
                .iter()
                .any(|t| matches!(t.dtype, GgmlType::IQ3_XXS | GgmlType::IQ3_S)),
        "standard IQ3 first-shard path required, not Q2 hybrid"
    );
    Ok(())
}

fn schedule(s: &v41_gguf::Schedule) -> Value {
    json!({"layers": s.layers, "kv_source_layers": s.kv_source_layers,
        "index_source_layers": s.index_source_layers, "compression_ratios": s.compression_ratios,
        "candidate_source_layer": s.candidate_source_layer})
}

pub(super) fn artifact(
    g: &crate::gguf::GgufFile,
    p: &v41_gguf::Plan,
    original: &Value,
) -> AResult<Value> {
    let stamps = g.revalidate_retained_shard_stamps()?;
    let shards = original["shards"]
        .as_array()
        .context("frozen shard stamps")?;
    ensure!(
        stamps.len() == shards.len() && stamps.len() == g.shards.len(),
        "shard inventory mismatch"
    );
    let mut actual_stamps = Vec::new();
    for ((stamp, shard), expected) in stamps.iter().zip(&g.shards).zip(shards) {
        let value = json!({"shard_idx": stamp.shard_idx, "supplied_path": stamp.path,
            "device": stamp.device, "inode": stamp.inode, "size": stamp.size,
            "mtime_sec": stamp.mtime_sec, "mtime_nsec": stamp.mtime_nsec,
            "ctime_sec": stamp.ctime_sec, "ctime_nsec": stamp.ctime_nsec,
            "tensor_data_start": shard.tensor_data_start, "alignment": shard.alignment});
        for (key, actual) in value.as_object().unwrap() {
            ensure!(
                expected[key] == *actual,
                "retained GGUF stamp/layout mismatch shard={} field={key}; stamp reuse requires the original files, not copied shards",
                stamp.shard_idx
            );
        }
        actual_stamps.push(value);
    }
    let tensors: Vec<_> = g.tensors.iter().map(|t| json!({"name": t.name, "dtype": format!("{:?}", t.dtype),
        "shape": t.shape, "shard_idx": t.shard_idx, "data_offset": t.data_offset, "n_bytes": t.n_bytes})).collect();
    let layout = json!({"metadata": g.model.metadata(), "tensors": tensors,
        "shards": actual_stamps.iter().map(|s| json!({"shard_idx": s["shard_idx"], "size": s["size"],
            "alignment": s["alignment"], "tensor_data_start": s["tensor_data_start"]})).collect::<Vec<_>>()});
    ensure!(
        archive::json_sha(&layout)? == original["layout_sha256"] && layout == original["layout"],
        "actual GGUF layout/metadata mismatch"
    );
    ensure!(
        original["validated_config_debug"] == format!("{:?}", p.config)
            && original["schedule"] == schedule(&p.schedule)
            && original["hash_profile_debug"] == format!("{:?}", p.engram_hash)
            && original["dtype_inventory_bytes"] == serde_json::to_value(&p.dtype_inventory)?
            && original["token_map_sha256_u32_le"]
                == archive::sha(bytemuck::cast_slice(&p.token_map))
            && original["raw_pad"] == p.raw_pad
            && original["compressed_pad"] == p.token_map[p.raw_pad as usize],
        "artifact execution identity mismatch"
    );
    ensure!(
        stamps == g.revalidate_retained_shard_stamps()?,
        "GGUF changed during stamp check"
    );
    Ok(json!(actual_stamps))
}

pub(super) fn frozen(v: &Value) -> AResult<()> {
    ensure!(
        v["runtime_source_commit"] == archive::HARNESS
            && v["original_runtime_pin"] == archive::RUNTIME
            && v["baseline_diff_sha256"]
                == "300702d2d97fcfc2efc38694847e8e19bb2a4f4cd1e67661e732e177e910e101",
        "frozen runtime/source diff mismatch"
    );
    let expected = [
        (
            "crates/qwen-llm/src/deepseek_v41/forward.rs",
            "d6026f611855834642ec47ac95be06b9339e299d4e32572635e8387009e5e3b5",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs",
            "0442dd8cdb3a00142cea120d3603c74f7f1db01716926ba8715a61bc661209f4",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs",
            "52965243096bed4d311aeffb82718afb9a6d8b28be48493186f2bf5162e9c090",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference.rs",
            "b6acb5546c563f9cee40ac40dc5d8eb0a713327a3b94fa901197e145ac8dcaef",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/hook.rs",
            "cfe266d7e00868469f636eaf5c28e384cd883fe55ff2894d5765e4a3ccd930fe",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/archive.rs",
            "3e6e66ef89c0a11b80f7b297ccd1a08d6c4ffd1f72180e09fadef7d11b44bd8a",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/plan.rs",
            "f562a44345e1cd652c8badb6ce4e36bf7fb46747a30c50e77c0258e27391ef33",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/input.rs",
            "3ca00a9ddbf727ba0996b2628adb0c132e9baa54d6c8cb0a814617f6052910de",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/provenance.rs",
            "7766a0fed0cbef2371c03de264c0477ab95ecd2a09ffcd2beb211b49b34fb6f7",
        ),
        (
            "Cargo.lock",
            "71089d1061368c463c1d7aee966004057ed6106a9101c2d49ca59930fab54a69",
        ),
    ];
    let compiled = v["compiled_sources"]
        .as_array()
        .context("frozen compiled sources")?;
    ensure!(
        compiled.len() == expected.len(),
        "frozen source inventory mismatch"
    );
    for (actual, (path, hash)) in compiled.iter().zip(expected) {
        ensure!(
            actual["path"] == path && actual["sha256"] == hash,
            "frozen compiled source mismatch: {path}"
        );
    }
    Ok(())
}

pub(super) fn source() -> AResult<(PathBuf, Value)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    macro_rules! embedded {
        ($($path:literal),+ $(,)?) => { &[$(($path, include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../", $path)).as_slice())),+] };
    }
    let sources: &[(&str, &[u8])] = embedded![
        "Cargo.lock",
        "Cargo.toml",
        "crates/qwen-llm/Cargo.toml",
        "crates/qwen-llm/build.rs",
        "crates/qwen-llm/src/deepseek_v41/forward.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced_plan.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/query_batch.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/prefill.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/bind.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/capture.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/engram_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/hc_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/memory.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/snapshot.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/token_map.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/error.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/profile.rs",
        "crates/qwen-llm/src/deepseek_v41/engram.rs",
        "crates/qwen-llm/src/deepseek_v41/gguf.rs",
        "crates/qwen-llm/src/deepseek_v41/gguf/metadata.rs",
        "crates/qwen-llm/src/deepseek_v41/gguf/names.rs",
        "crates/qwen-llm/src/deepseek_v41/gguf/engram.rs",
        "crates/qwen-llm/src/deepseek_v41/gguf/storage.rs",
        "crates/qwen-llm/src/gguf.rs",
        "crates/qwen-llm/src/checkpoint_identity.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/archive.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/archive/tests.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/metrics.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/provenance.rs",
        "kernels/deepseek_v41_attention.metal",
        "kernels/deepseek_v41_ffn.metal",
        "kernels/deepseek_v41_engram.metal",
    ];
    let mut compiled = Vec::new();
    for &(path, bytes) in sources {
        let hash = archive::sha(bytes);
        ensure!(
            archive::file_sha(&mut archive::open(&root.join(path))?)? == hash,
            "binary/source mismatch: {path}; rebuild after wiring or editing"
        );
        compiled.push(json!({"path": path, "sha256": hash}));
    }
    let git = |args: &[&str]| -> AResult<String> {
        let o = Command::new("git").current_dir(&root).args(args).output()?;
        ensure!(
            o.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        Ok(String::from_utf8(o.stdout)?)
    };
    let exe = std::env::current_exe()?;
    let value = json!({"runtime_source_commit": git(&["rev-parse", "HEAD"])?.trim(),
        "runtime_source_tree": git(&["rev-parse", "HEAD^{tree}"])?.trim(),
        "worktree_status": git(&["status", "--porcelain=v1"])?, "tracked_diff_sha256": archive::sha(git(&["diff", "--binary", "HEAD"])?.as_bytes()),
        "compiled_sources": compiled, "binary_path": exe, "binary_sha256": archive::file_sha(&mut archive::open(&exe)?)?,
        "build": {"debug_assertions": cfg!(debug_assertions), "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "package_version": env!("CARGO_PKG_VERSION"), "rustflags_at_compile": option_env!("RUSTFLAGS"),
            "encoded_rustflags_at_compile": option_env!("CARGO_ENCODED_RUSTFLAGS"),
            "compiler_attestation": "not build-attested; fixed included source set checked against disk before/after; actual executing binary hashed"},
        "metallib_build": {"compiler_and_sdk": crate::metallib_build_identity(), "product_target": crate::PRODUCT_METALLIB_TARGET,
            "tensor_target": crate::TENSOR_METALLIB_TARGET, "tensor_tiers": crate::TENSOR_METALLIB_TIERS,
            "product_sha256": archive::sha(crate::KERNELS_METALLIB), "tensor_sha256": archive::sha(crate::KERNELS_TENSOR_METALLIB),
            "diagnostic_features": crate::enabled_diagnostic_features()}});
    Ok((root, value))
}
