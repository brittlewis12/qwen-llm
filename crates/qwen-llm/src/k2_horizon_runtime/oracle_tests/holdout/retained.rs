//! Replay existing independent outputs; never execute or regenerate the reference.
use super::*;

// Original materialized v2 evidence root, pinned before any online replay.
const MANIFEST_SHA256: &str = "191387561440961677bbd64fad302627a1f5dc428fafe79eeec3294f53d1e3af";
const REFERENCES_SHA256: &str = "bbc0e26e0d18e0058cf4b6fef797f45cbb00555916ce582c572860099588b8ee";

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn validate_manifest(
    manifest: &serde_json::Value,
    p: &serde_json::Value,
    inputs: &serde_json::Value,
    binary_hash: &str,
) {
    assert_eq!(manifest["policy"], *p);
    assert_eq!(manifest["policy_sha256"], v2::SHA256);
    assert_eq!(manifest["fixture_freeze_commit"], "300c8dcb");
    assert_eq!(manifest["inputs"], *inputs);
    assert_eq!(manifest["reference"], reference_identity().trim());
    assert_eq!(manifest["reference_binary_sha256"], binary_hash);
    assert_eq!(manifest["reference_cache"], "F16");
    assert_eq!(manifest["reference_allocated_capacity"], 256);
    assert_eq!(manifest["native_attention"], "materialized");
    assert_eq!(manifest["native_cache"], "F16");
    assert_eq!(manifest["native_capacity"], 256);
    assert_eq!(manifest["metal_api_validation"], "1");
    assert_eq!(manifest["public_cap_promoted"], false);
}

fn checked_file(path: PathBuf, digest: &serde_json::Value) -> PathBuf {
    assert_eq!(file_sha256(&path), digest.as_str().unwrap());
    path
}

fn checked_rows(path: &Path, base: u32, tokens: &[u32], greedy: bool) {
    validate_reference_log(&String::from_utf8_lossy(
        &fs::read(path.with_extension("log")).unwrap(),
    ));
    let mut rows = OracleRows::new(
        BufReader::new(fs::File::open(path).unwrap()),
        base,
        tokens,
        250624,
    );
    let mut next = None;
    for (index, &token) in tokens.iter().enumerate() {
        if greedy && index >= 241 {
            assert_eq!(next, Some(token), "retained trajectory is not argmax");
        }
        next = Some(runner::argmax(&rows.row()));
    }
    rows.finish();
}

fn load(
    root: &Path,
    p: &serde_json::Value,
    cases: &[(String, u32, Vec<u32>, bool)],
    binary: &Path,
) -> Vec<runner::ReferenceCase> {
    assert_eq!(file_sha256(&root.join("manifest.json")), MANIFEST_SHA256);
    assert_eq!(
        file_sha256(&root.join("references.json")),
        REFERENCES_SHA256
    );
    let manifest = read_json(&root.join("manifest.json"));
    validate_manifest(&manifest, p, &json!(cases), &file_sha256(binary));
    let records = read_json(&root.join("references.json"));
    let records = records.as_array().unwrap();
    assert_eq!(records.len(), cases.len());
    let mut references = Vec::new();
    for ((name, base, tokens, selected), record) in cases.iter().zip(records) {
        assert_eq!(record["corpus"], *name);
        assert_eq!(record["base"], *base);
        let directory = root.join(format!("{name}-{base}"));
        let filename = format!("reference-{base}.f32");
        let logits = checked_file(
            directory.join("ordinary").join(&filename),
            &record["ordinary_sha256"],
        );
        assert_eq!(
            logits.canonicalize().unwrap(),
            Path::new(record["ordinary"].as_str().unwrap())
                .canonicalize()
                .unwrap()
        );
        checked_rows(&logits, *base, tokens, false);
        let mut captures = None;
        let mut greedy = None;
        if *selected {
            assert_eq!(record["capture_noninterference_bitwise"], true);
            let traced = directory.join("captured").join(&filename);
            checked_rows(&traced, *base, tokens, false);
            let mut ordinary = OracleRows::new(
                BufReader::new(fs::File::open(&logits).unwrap()),
                *base,
                tokens,
                250624,
            );
            let mut captured = OracleRows::new(
                BufReader::new(fs::File::open(&traced).unwrap()),
                *base,
                tokens,
                250624,
            );
            for _ in tokens {
                assert!(
                    runner::bitwise_equal(&ordinary.row(), &captured.row()),
                    "retained capture changed logits"
                );
            }
            ordinary.finish();
            captured.finish();
            let layers = checked_file(
                traced.with_extension("f32.layers"),
                &record["capture_sha256"],
            );
            captures = Some(decode_layers(
                &fs::read(layers).unwrap(),
                base + 255,
                tokens[255],
            ));
            let trajectory = checked_file(
                directory.join("greedy").join(&filename),
                &record["greedy_sha256"],
            );
            let ids_file = checked_file(
                trajectory.with_extension("f32.tokens"),
                &record["greedy_tokens_sha256"],
            );
            let ids = runner::generated_ids(&fs::read(ids_file).unwrap(), *base, &tokens[..241]);
            assert_eq!(record["greedy_tokens"], json!(ids));
            checked_rows(&trajectory, *base, &ids, true);
            greedy = Some((trajectory, ids));
        } else {
            assert!(record.get("capture_sha256").is_none());
            assert!(record.get("greedy_sha256").is_none());
        }
        references.push(runner::ReferenceCase {
            logits,
            captures,
            greedy,
        });
    }
    references
}

#[test]
#[ignore = "GPU online replay of retained v2 IFM outputs; production lease; no reference child or new logit dumps"]
fn gpu_online_against_retained_independent_v2_reference() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let path = PathBuf::from(std::env::var("K2_GGUF").expect("K2_GGUF"));
    let binary = PathBuf::from(
        std::env::var("K2_LLAMA_ORACLE").expect("K2_LLAMA_ORACLE (hash only, never executed)"),
    );
    let root = PathBuf::from(std::env::var("K2_RETAINED_V2").expect("K2_RETAINED_V2"))
        .canonicalize()
        .unwrap();
    let source = GgufFile::open(&path).unwrap();
    let p = v2::policy();
    let cases = inputs_for(&source, &p);
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let manifest_hash = file_sha256(&root.join("manifest.json"));
    let references_hash = file_sha256(&root.join("references.json"));
    let files = load(&root, &p, &cases, &binary);
    assert_eq!(file_sha256(&root.join("manifest.json")), manifest_hash);
    assert_eq!(file_sha256(&root.join("references.json")), references_hash);
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!(
            "k2-online-retained-v2-{}-{stamp}",
            std::process::id()
        ));
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "claim":"online_backend_regression_against_retained_independent_reference_not_new_holdout",
        "policy":p,"policy_sha256":v2::SHA256,"fixture_freeze_commit":"300c8dcb",
        "retained_directory":root,"retained_manifest_sha256":manifest_hash,
        "retained_references_sha256":references_hash,"reference":reference_identity().trim(),
        "reference_binary_sha256":file_sha256(&binary),"reference_children_executed":0,
        "capture_noninterference_rechecked":true,"reference_cache":"F16","reference_allocated_capacity":256,
        "reference_log_binding":"pinned-source exact content markers; log bytes were not digest-bound by original producer",
        "model":path,"inputs":cases,"native_capacity":256,"native_cache":"F16","native_attention":"online_experimental",
        "native_test_binary_sha256":file_sha256(&std::env::current_exe().unwrap()),
        "native_metallib_sha256":native_kernel_identity(),"native_sources":native_source_identity(),
        "host_build_note":std::env::var("K2_HOST_BUILD_NOTE").ok(),"metal_api_validation":"1",
        "public_cap_promoted":false,"public_default_changed":false,"performance_claim":false,
    })).unwrap()).unwrap();
    eprintln!("retained online replay artifacts: {}", directory.display());
    runner::evaluate(
        p,
        v2::SHA256,
        &source,
        &cases,
        files,
        &directory,
        AttentionBackend::OnlineExperimental,
        v2::row_failures,
    );
}

#[test]
fn retained_manifest_rejects_policy_identity_backend_and_mode_drift() {
    let p = v2::policy();
    let inputs = json!(["test inputs"]);
    let hash = "a".repeat(64);
    let valid = json!({"policy":p,"policy_sha256":v2::SHA256,"fixture_freeze_commit":"300c8dcb",
        "inputs":inputs,"reference":reference_identity().trim(),"reference_binary_sha256":hash,
        "reference_cache":"F16","reference_allocated_capacity":256,"native_attention":"materialized",
        "native_cache":"F16","native_capacity":256,"metal_api_validation":"1","public_cap_promoted":false});
    validate_manifest(&valid, &p, &inputs, &hash);
    for key in valid.as_object().unwrap().keys() {
        let mut bad = valid.clone();
        bad[key] = serde_json::Value::Null;
        assert!(
            std::panic::catch_unwind(|| validate_manifest(&bad, &p, &inputs, &hash)).is_err(),
            "{key}"
        );
    }
    let mut wrong_model = p.clone();
    wrong_model["model_sha256"] = json!("b".repeat(64));
    assert!(
        std::panic::catch_unwind(|| validate_manifest(&valid, &wrong_model, &inputs, &hash))
            .is_err()
    );
}
