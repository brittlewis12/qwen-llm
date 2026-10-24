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

const FROZEN_BASELINE_DIFF_SHA256: &str =
    "1b2bf6b0384435d7cb551f4df7444beb52da51ed62bd52f3e2626bf0f1c675c5";

fn frozen_compiled_sources() -> &'static [(&'static str, &'static str)] {
    &[
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
            "b59e23f8120a14dcec393cfb28c7efbc834270cb868ee31b93cc8072644f2abe",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/hook.rs",
            "53357ae1b9b6fc8e69a8ae47dcf2802e3fc3602ead574445f67855dbaea1e9cb",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/archive.rs",
            "ccddb16aeffab7270fc0615645a09068af2a3920d3dcd34427434448bd5083ba",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/plan.rs",
            "773b2ce7acdc99ae5e11472692f8f018ba5999eed92d2507461c6843910de95f",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/input.rs",
            "3ca00a9ddbf727ba0996b2628adb0c132e9baa54d6c8cb0a814617f6052910de",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/provenance.rs",
            "f89dd2b18bde1713d5031ab573480ef5d6582194eca92d33bd18febd3bbe6ef1",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/copy_extent.rs",
            "f03cd611c557abfa6f55f575cb3e5bc2b1b5f046b09d8e54dd55b081d2d36624",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/smoke.rs",
            "cf86d4cda5d44a9abaad7169ce9e09fab2b6059a70777335bd9f8bab1024859e",
        ),
        (
            "crates/qwen-llm/src/deepseek_v41/forward/ced_reference/artifact_cache.rs",
            "ac335f94f1b1795afe670a3f5dc7039c347ccefc2210944f7234a2a4571d29f8",
        ),
        (
            "Cargo.lock",
            "71089d1061368c463c1d7aee966004057ed6106a9101c2d49ca59930fab54a69",
        ),
    ]
}

pub(super) fn frozen(v: &Value) -> AResult<()> {
    ensure!(
        v["runtime_source_commit"] == archive::HARNESS
            && v["original_runtime_pin"] == archive::RUNTIME
            && v["baseline_diff_sha256"] == FROZEN_BASELINE_DIFF_SHA256,
        "frozen runtime/source diff mismatch"
    );
    let expected = frozen_compiled_sources();
    let compiled = v["compiled_sources"]
        .as_array()
        .context("frozen compiled sources")?;
    ensure!(
        compiled.len() == expected.len(),
        "frozen source inventory mismatch"
    );
    for (actual, &(path, hash)) in compiled.iter().zip(expected) {
        ensure!(
            actual["path"] == path && actual["sha256"] == hash,
            "frozen compiled source mismatch: {path}"
        );
    }
    Ok(())
}

#[test]
fn frozen_reference_requires_complete_reviewed_harness_identity() {
    let sources = frozen_compiled_sources()
        .iter()
        .map(|&(path, hash)| json!({"path": path, "sha256": hash}))
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 13);
    let good = json!({"runtime_source_commit": archive::HARNESS,
        "original_runtime_pin": archive::RUNTIME,
        "baseline_diff_sha256": FROZEN_BASELINE_DIFF_SHA256, "compiled_sources": sources});
    frozen(&good).unwrap();
    for field in [
        "runtime_source_commit",
        "original_runtime_pin",
        "baseline_diff_sha256",
    ] {
        let mut bad = good.clone();
        bad[field] = json!("unreviewed");
        assert!(frozen(&bad).is_err(), "accepted {field} substitution");
    }
    for i in 0..sources.len() {
        for field in ["path", "sha256"] {
            let mut bad = good.clone();
            bad["compiled_sources"][i][field] = json!("unreviewed");
            assert!(frozen(&bad).is_err(), "accepted source {i} {field}");
        }
    }
    let mut bad = good.clone();
    bad["compiled_sources"].as_array_mut().unwrap().pop();
    assert!(frozen(&bad).is_err());
    bad = good;
    bad["compiled_sources"].as_array_mut().unwrap().reverse();
    assert!(frozen(&bad).is_err());
}

pub(super) fn batching_compiled_sources() -> &'static [(&'static str, &'static [u8])] {
    macro_rules! embedded {
        ($($path:literal),+ $(,)?) => { &[$(($path, include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../", $path)).as_slice())),+] };
    }
    embedded![
        "crates/qwen-llm/src/deepseek_v41/forward/ced_plan/batches.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced/batches.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced/batches/tests.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/tests.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced/tests/reuse.rs",
        "crates/qwen-llm/src/metal/elementwise.rs",
        "kernels/elementwise.metal",
    ]
}

pub(super) fn checked_compiled_source(root: &Path, path: &str, bytes: &[u8]) -> AResult<Value> {
    let hash = archive::sha(bytes);
    ensure!(
        archive::file_sha(&mut archive::open(&root.join(path))?)? == hash,
        "binary/source mismatch: {path}; rebuild after wiring or editing"
    );
    Ok(json!({"path": path, "sha256": hash}))
}

pub(super) fn checked_batching_sources(root: &Path) -> AResult<Vec<Value>> {
    batching_compiled_sources()
        .iter()
        .map(|&(path, bytes)| checked_compiled_source(root, path, bytes))
        .collect()
}

fn compiled_sources() -> &'static [(&'static str, &'static [u8])] {
    macro_rules! embedded {
        ($($path:literal),+ $(,)?) => { &[$(($path, include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../", $path)).as_slice())),+] };
    }
    const SOURCES: &[(&str, &[u8])] = embedded![
        "Cargo.lock",
        "Cargo.toml",
        "crates/qwen-llm/Cargo.toml",
        "crates/qwen-llm/build.rs",
        "crates/qwen-llm/src/deepseek_v41/forward.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/chunk_geometry.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/preflight.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/binding.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced_plan.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/gpu_step/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/ced.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/query_batch.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu/tests/arithmetic.rs",
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
        "crates/qwen-llm/src/metal/moe_grouped_generic.rs",
        "crates/qwen-llm/src/metal/moe_grouped_generic/tensor.rs",
        "crates/qwen-llm/src/metal/moe_grouped_generic/tests.rs",
        "crates/qwen-llm/src/metal/context.rs",
        "crates/qwen-llm/src/metal/census.rs",
        "crates/qwen-llm/src/metal/host_budget.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/archive.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/archive/tests.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/metrics.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/provenance.rs",
        "crates/qwen-llm/src/deepseek_v41/forward/ced/qualification/current_pair.rs",
        "kernels/deepseek_v41_attention.metal",
        "kernels/deepseek_v41_ffn.metal",
        "kernels/deepseek_v41_engram.metal",
        "kernels/tensor/moe_grouped_slots_tensor.metal",
        "kernels/quant_tiles.h",
    ];
    SOURCES
}

fn checked_compiled_sources(root: &Path) -> AResult<Vec<Value>> {
    let mut compiled = Vec::new();
    for &(path, bytes) in compiled_sources() {
        compiled.push(checked_compiled_source(&root, path, bytes)?);
    }
    compiled.extend(checked_batching_sources(root)?);
    Ok(compiled)
}

#[test]
fn active_compiled_inventory_matches_checkout() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    checked_compiled_sources(&root).unwrap();
}

#[test]
fn compiled_geometry_inventory_rejects_helper_only_edits_before_import() {
    let helper = "crates/qwen-llm/src/deepseek_v41/forward/chunk_geometry.rs";
    let sources = compiled_sources();
    let embedded = sources.iter().find(|(p, _)| *p == helper).unwrap().1;
    assert_eq!(embedded, include_bytes!("../../chunk_geometry.rs"));
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "qwen-geometry-inventory-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    for &(path, bytes) in sources.iter().chain(batching_compiled_sources()) {
        let destination = root.join(path);
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        std::fs::write(destination, bytes).unwrap();
    }
    assert_eq!(
        checked_compiled_sources(&root).unwrap().len(),
        sources.len() + batching_compiled_sources().len()
    );
    let mut changed = embedded.to_vec();
    changed.extend_from_slice(b"\n// helper-only freshness probe\n");
    std::fs::write(root.join(helper), changed).unwrap();
    let refusal = checked_compiled_sources(&root).unwrap_err().to_string();
    assert!(
        refusal.contains(&format!("binary/source mismatch: {helper}")),
        "{refusal}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

pub(super) fn source() -> AResult<(PathBuf, Value)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let compiled = checked_compiled_sources(&root)?;
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

#[test]
fn every_batching_source_rejects_a_stale_body_in_both_qualification_inventories() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "qwen-batch-freshness-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    for &(path, compiled) in batching_compiled_sources() {
        let file = root.join(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, compiled).unwrap();
    }
    for &(path, compiled) in batching_compiled_sources() {
        let file = root.join(path);
        assert!(checked_batching_sources(&root).is_ok());
        assert!(super::current_pair::checked_batching_sources(&root).is_ok());
        let mut changed = compiled.to_vec();
        changed.extend_from_slice(b"\n// freshness regression edit\n");
        std::fs::write(&file, changed).unwrap();
        for error in [
            checked_batching_sources(&root).unwrap_err(),
            super::current_pair::checked_batching_sources(&root).unwrap_err(),
        ] {
            assert!(
                error
                    .to_string()
                    .contains(&format!("binary/source mismatch: {path}")),
                "{error}"
            );
        }
        std::fs::write(&file, compiled).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}
