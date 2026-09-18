use super::*;
use crate::tokenizer::NativeTokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn reference_identity() -> String {
    let wrapper = Sha256::digest(include_bytes!("../../../../scripts/reference/k2/main.cpp"));
    let cmake = Sha256::digest(include_bytes!(
        "../../../../scripts/reference/k2/CMakeLists.txt"
    ));
    format!(
        "42adf019f76013dac873b5b43950d54d5ab27216 F16-KV serial flash=off wrapper={wrapper:x} cmake={cmake:x}\n"
    )
}

fn file_sha256(path: &Path) -> String {
    // Debug-build Rust hashing of a 9.57 GB artifact is prohibitively slow.
    let output = Command::new("/usr/bin/shasum")
        .args(["-a", "256", "--"])
        .arg(path)
        .output()
        .unwrap();
    assert!(output.status.success(), "artifact hash command failed");
    let text = String::from_utf8(output.stdout).unwrap();
    let digest = text.split_whitespace().next().unwrap();
    assert_eq!(digest.len(), 64);
    assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()));
    digest.to_owned()
}

fn validate_reference_log(log: &str) {
    let log = log
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>();
    for required in [
        "load_tensors: offloaded 37/37 layers to GPU",
        "llama_context: n_ctx = 256",
        "llama_context: n_batch = 1",
        "llama_context: n_ubatch = 1",
        "llama_context: flash_attn = disabled",
        "llama_context: freq_base = 10000000.0",
        "llama_context: freq_scale = 1",
        "llama_kv_cache: size = 36.00 MiB ( 256 cells, 36 layers, 1/1 seqs), K (f16): 18.00 MiB, V (f16): 18.00 MiB",
        "ggml_metal_device_init: GPU name: MTL0 (Apple M4 Max)",
    ] {
        assert!(
            log.iter().any(|line| line == required),
            "oracle runtime record missing {required}"
        );
    }
    for layer in 0..36 {
        assert!(
            log.iter()
                .any(|line| line == &format!("llama_kv_cache: layer {layer}: dev = MTL0")),
            "cache layer {layer} is not on MTL0"
        );
    }
}

fn decode_rows(bytes: &[u8], base: u32, tokens: &[u32], vocab: usize) -> Vec<Vec<f32>> {
    let row_bytes = 8 + 4 * vocab;
    assert_eq!(
        bytes.len(),
        24 + tokens.len() * row_bytes,
        "oracle output extent"
    );
    assert_eq!(&bytes[..8], b"K2REF001");
    let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(word(8) as usize, vocab);
    assert_eq!(word(12) as usize, tokens.len());
    assert_eq!(word(16), base);
    assert_eq!(word(20), 16);
    tokens
        .iter()
        .enumerate()
        .map(|(row, &id)| {
            let start = 24 + row * row_bytes;
            assert_eq!(word(start), base + row as u32);
            assert_eq!(word(start + 4), id);
            (0..vocab)
                .map(|i| {
                    let value = f32::from_bits(word(start + 8 + i * 4));
                    assert!(value.is_finite(), "nonfinite oracle output");
                    value
                })
                .collect()
        })
        .collect()
}

fn oracle_rows(
    binary: &Path,
    model: &Path,
    directory: &Path,
    base: u32,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    assert!(!tokens.is_empty() && tokens.len() <= 32);
    let output = directory.join(format!("reference-{base}.f32"));
    let log = directory.join(format!("reference-{base}.log"));
    let log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(log)
        .unwrap();
    let status = Command::new(binary)
        .arg(model)
        .arg(&output)
        .arg(base.to_string())
        .args(tokens.iter().map(u32::to_string))
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .status()
        .unwrap();
    assert!(
        status.success(),
        "oracle failed; inspect {}",
        directory.display()
    );
    validate_reference_log(&String::from_utf8_lossy(
        &fs::read(directory.join(format!("reference-{base}.log"))).unwrap(),
    ));
    let expected = 24 + tokens.len() * (8 + 250624 * 4);
    assert_eq!(fs::metadata(&output).unwrap().len(), expected as u64);
    decode_rows(&fs::read(output).unwrap(), base, tokens, 250624)
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    let mut squared = 0.0_f64;
    let mut maximum = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let (a, b) = (f64::from(a), f64::from(b));
        squared += (a - b).powi(2);
        maximum = maximum.max((a - b).abs());
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    json!({"max_abs": maximum, "rmse": (squared / actual.len() as f64).sqrt(),
        "cosine": dot/(aa*bb).sqrt(), "actual_top1": top(actual), "reference_top1": top(expected)})
}

#[test]
fn oracle_binary_protocol_binds_coordinates_and_exact_rows() {
    let mut bytes = b"K2REF001".to_vec();
    for word in [2, 1, 37, 16, 37, 19, 1_f32.to_bits(), (-2_f32).to_bits()] {
        bytes.extend(word.to_le_bytes());
    }
    assert_eq!(decode_rows(&bytes, 37, &[19], 2), vec![vec![1.0, -2.0]]);
    for offset in [8, 12, 16, 20, 24, 28] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(std::panic::catch_unwind(|| decode_rows(&corrupt, 37, &[19], 2)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| decode_rows(&bytes[..bytes.len() - 1], 37, &[19], 2)).is_err()
    );
}

#[test]
fn oracle_runtime_log_rejects_mode_and_backend_drift() {
    let mut log = concat!(
        "load_tensors: offloaded 37/37 layers to GPU\n",
        "llama_context: n_ctx = 256\nllama_context: n_batch = 1\nllama_context: n_ubatch = 1\n",
        "llama_context: flash_attn = disabled\nllama_context: freq_base = 10000000.0\n",
        "llama_context: freq_scale = 1\n",
        "llama_kv_cache: size = 36.00 MiB ( 256 cells, 36 layers, 1/1 seqs), K (f16): 18.00 MiB, V (f16): 18.00 MiB\n",
        "ggml_metal_device_init: GPU name: MTL0 (Apple M4 Max)\n",
    ).to_owned();
    for layer in 0..36 {
        log.push_str(&format!("llama_kv_cache: layer {layer}: dev = MTL0\n"));
    }
    validate_reference_log(&log);
    for (from, to) in [
        ("freq_scale = 1", "freq_scale = 10"),
        ("37/37", "36/37"),
        ("n_ctx = 256", "n_ctx = 2560"),
        ("K (f16)", "K (f32)"),
        ("layer 35: dev = MTL0", "layer 35: dev = CPU"),
        ("flash_attn = disabled", "flash_attn = enabled"),
    ] {
        let corrupt = log.replace(from, to);
        assert!(std::panic::catch_unwind(|| validate_reference_log(&corrupt)).is_err());
    }
}

#[test]
#[ignore = "GPU/checkpoint correctness; requires K2_GGUF and pinned K2_LLAMA_ORACLE; holds production lease"]
fn gpu_checkpoint_logits_match_independent_ifm_fork() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(
        crate::metal::mat_vec_q8_0_lcpp_enabled(),
        "qualification requires the default Q8 matvec lineage"
    );
    let path = PathBuf::from(std::env::var("K2_GGUF").expect("K2_GGUF"));
    let binary = PathBuf::from(std::env::var("K2_LLAMA_ORACLE").expect("K2_LLAMA_ORACLE"));
    let identity = Command::new(&binary).arg("--identity").output().unwrap();
    assert!(identity.status.success());
    assert_eq!(
        String::from_utf8(identity.stdout).unwrap(),
        reference_identity()
    );
    let source = GgufFile::open(&path).unwrap();
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    let model_sha256 = file_sha256(&path);
    assert_eq!(
        model_sha256, "5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf",
        "this qualification corpus targets the pinned Q8 artifact, not runtime admission"
    );
    let tokenizer = NativeTokenizer::from_gguf(&source).unwrap();
    let text = tokenizer
        .encode("The capital of France is", true)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .collect::<Vec<_>>();
    let code = tokenizer.encode("fn fibonacci(n: u32) -> u32 {\n    if n < 2 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }\n}\n", true).unwrap()
        .into_iter().take(32).map(|id| u32::try_from(id).unwrap()).collect::<Vec<_>>();
    assert_eq!(code.len(), 32);
    let cases = [(0, vec![0, 42, 17, 19]), (37, text), (128, code)];
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-oracle-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
        "reference": reference_identity().trim(), "reference_binary_sha256": file_sha256(&binary),
            "model": path, "model_sha256": model_sha256, "inputs": cases,
            "requested_reference_capacity": 32, "reference_allocated_capacity": 256,
            "native_capacity": 32, "cache": "F16", "native_q8_matvec": "lcpp",
            "metal_api_validation": std::env::var("MTL_DEBUG_LAYER").ok(),
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!("oracle artifacts: {}", directory.display());
    // The parent holds the production lease while each standalone reference
    // process runs. No native context/weights exist until those children exit.
    let references = cases
        .iter()
        .map(|(base, tokens)| oracle_rows(&binary, &path, &directory, *base, tokens))
        .collect::<Vec<_>>();
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 32).unwrap();
    let mut reports = Vec::new();
    for ((base, tokens), rows) in cases.iter().zip(references) {
        let mut session = model.create_session(*base).unwrap();
        for (i, (&token, reference)) in tokens.iter().zip(rows).enumerate() {
            let actual = session.append(&[token]).unwrap();
            let metrics = error_metrics(&actual, &reference);
            eprintln!("base={base} step={i} {metrics}");
            reports.push(json!({"base": base, "step": i, "token": token, "metrics": metrics}));
        }
    }
    fs::write(
        directory.join("metrics.json"),
        serde_json::to_vec_pretty(&json!({
            "reference": reference_identity().trim(), "model": path, "cases": reports,
        }))
        .unwrap(),
    )
    .unwrap();
    for report in reports {
        let m = &report["metrics"];
        assert_eq!(m["actual_top1"], m["reference_top1"], "{report}");
        assert!(m["max_abs"].as_f64().unwrap() < 0.005, "{report}");
        assert!(m["rmse"].as_f64().unwrap() < 0.001, "{report}");
        assert!(m["cosine"].as_f64().unwrap() > 0.9999999, "{report}");
    }
}
