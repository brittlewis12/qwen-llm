use super::*;
use crate::tokenizer::NativeTokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

mod attention_probe;
mod cache_precision;
mod holdout;
mod probability;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceCache {
    F16,
    F32,
}

#[derive(Clone, Copy)]
enum ReferenceRun {
    F16,
    F32,
    Capture,
    Greedy15,
}

impl ReferenceCache {
    fn bits(self) -> u32 {
        match self {
            Self::F16 => 16,
            Self::F32 => 32,
        }
    }
    fn cache_log(self) -> &'static str {
        match self {
            Self::F16 => {
                "llama_kv_cache: size = 36.00 MiB ( 256 cells, 36 layers, 1/1 seqs), K (f16): 18.00 MiB, V (f16): 18.00 MiB"
            }
            Self::F32 => {
                "llama_kv_cache: size = 72.00 MiB ( 256 cells, 36 layers, 1/1 seqs), K (f32): 36.00 MiB, V (f32): 36.00 MiB"
            }
        }
    }
}

fn reference_identity() -> String {
    let wrapper = Sha256::digest(include_bytes!("../../../../scripts/reference/k2/main.cpp"));
    let cmake = Sha256::digest(include_bytes!(
        "../../../../scripts/reference/k2/CMakeLists.txt"
    ));
    format!(
        "42adf019f76013dac873b5b43950d54d5ab27216 default=F16-KV diagnostic=F32-KV greedy=15 serial flash=off wrapper={wrapper:x} cmake={cmake:x}\n"
    )
}

fn native_kernel_identity() -> String {
    format!("{:x}", Sha256::digest(crate::KERNELS_METALLIB))
}

fn native_source_identity() -> serde_json::Value {
    json!({
        "runtime_sha256":format!("{:x}", Sha256::digest(include_bytes!("../k2_horizon_runtime.rs"))),
        "primitives_sha256":format!("{:x}", Sha256::digest(include_bytes!("../k2_horizon_metal.rs"))),
        "default_attention":format!("{DEFAULT_ATTENTION_BACKEND:?}"),
        "packed_sha256":format!("{:x}", Sha256::digest(include_bytes!("packed.rs"))),
        "state_sha256":format!("{:x}", Sha256::digest(include_bytes!("state.rs"))),
        "compact_cache_sha256":format!("{:x}", Sha256::digest(include_bytes!("../k2_horizon_metal/compact.rs"))),
        "cache_plan_sha256":format!("{:x}", Sha256::digest(include_bytes!("../k2_horizon_plan.rs"))),
        "residency_sha256":format!("{:x}", Sha256::digest(include_bytes!("residency.rs"))),
    })
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
    validate_reference_log_cache(log, ReferenceCache::F16);
}

fn validate_reference_log_cache(log: &str, cache: ReferenceCache) {
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
        cache.cache_log(),
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
    let mut reader = OracleRows::new(Cursor::new(bytes), base, tokens, vocab);
    let rows = (0..tokens.len()).map(|_| reader.row()).collect();
    reader.finish();
    rows
}

struct OracleRows<'a, R> {
    reader: R,
    tokens: &'a [u32],
    base: u32,
    vocab: usize,
    index: usize,
}

impl<'a, R: Read> OracleRows<'a, R> {
    fn new(reader: R, base: u32, tokens: &'a [u32], vocab: usize) -> Self {
        Self::with_cache(reader, base, tokens, vocab, ReferenceCache::F16)
    }

    fn with_cache(
        mut reader: R,
        base: u32,
        tokens: &'a [u32],
        vocab: usize,
        cache: ReferenceCache,
    ) -> Self {
        assert!((1..=256).contains(&tokens.len()));
        assert!((1..=250624).contains(&vocab));
        assert!(base.checked_add(tokens.len() as u32 - 1).is_some());
        let mut header = [0; 24];
        reader
            .read_exact(&mut header)
            .expect("oracle header extent");
        assert_eq!(&header[..8], b"K2REF001");
        let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(8) as usize, vocab);
        assert_eq!(word(12) as usize, tokens.len());
        assert_eq!(word(16), base);
        assert_eq!(word(20), cache.bits());
        Self {
            reader,
            tokens,
            base,
            vocab,
            index: 0,
        }
    }

    fn row(&mut self) -> Vec<f32> {
        assert!(self.index < self.tokens.len(), "extra oracle row request");
        let mut bytes = vec![0; 8 + 4 * self.vocab];
        self.reader
            .read_exact(&mut bytes)
            .expect("truncated oracle row");
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(0), self.base + self.index as u32);
        assert_eq!(word(4), self.tokens[self.index]);
        let values = (0..self.vocab)
            .map(|i| {
                let value = f32::from_bits(word(8 + i * 4));
                assert!(value.is_finite(), "nonfinite oracle output");
                value
            })
            .collect();
        self.index += 1;
        values
    }

    fn finish(mut self) {
        assert_eq!(self.index, self.tokens.len(), "unread oracle rows");
        let mut trailing = [0u8; 1];
        assert_eq!(
            self.reader.read(&mut trailing).unwrap(),
            0,
            "trailing oracle bytes"
        );
    }
}

fn oracle_rows(
    binary: &Path,
    model: &Path,
    directory: &Path,
    base: u32,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let output = run_oracle(binary, model, directory, base, tokens);
    decode_rows(&fs::read(output).unwrap(), base, tokens, 250624)
}

fn run_oracle(binary: &Path, model: &Path, directory: &Path, base: u32, tokens: &[u32]) -> PathBuf {
    run_oracle_mode(binary, model, directory, base, tokens, false)
}

fn run_oracle_mode(
    binary: &Path,
    model: &Path,
    directory: &Path,
    base: u32,
    tokens: &[u32],
    capture_last: bool,
) -> PathBuf {
    run_oracle_cache(
        binary,
        model,
        directory,
        base,
        tokens,
        capture_last,
        ReferenceCache::F16,
    )
}

fn run_oracle_cache(
    binary: &Path,
    model: &Path,
    directory: &Path,
    base: u32,
    tokens: &[u32],
    capture_last: bool,
    cache: ReferenceCache,
) -> PathBuf {
    assert!(!capture_last || cache == ReferenceCache::F16);
    let mode = if capture_last {
        ReferenceRun::Capture
    } else {
        match cache {
            ReferenceCache::F16 => ReferenceRun::F16,
            ReferenceCache::F32 => ReferenceRun::F32,
        }
    };
    run_oracle_command(binary, model, directory, base, tokens, mode)
}

fn run_oracle_command(
    binary: &Path,
    model: &Path,
    directory: &Path,
    base: u32,
    tokens: &[u32],
    mode: ReferenceRun,
) -> PathBuf {
    assert!(!tokens.is_empty() && tokens.len() <= 256);
    let (flag, cache, count) = match mode {
        ReferenceRun::F16 => (None, ReferenceCache::F16, tokens.len()),
        ReferenceRun::F32 => (Some("--f32-kv"), ReferenceCache::F32, tokens.len()),
        ReferenceRun::Capture => (Some("--capture-last"), ReferenceCache::F16, tokens.len()),
        ReferenceRun::Greedy15 => (Some("--greedy-15"), ReferenceCache::F16, tokens.len() + 15),
    };
    assert!(count <= 256);
    let output = directory.join(format!("reference-{base}.f32"));
    let log = directory.join(format!("reference-{base}.log"));
    let log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(log)
        .unwrap();
    let mut command = Command::new(binary);
    if let Some(flag) = flag {
        command.arg(flag);
    }
    let status = command
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
    validate_reference_log_cache(
        &String::from_utf8_lossy(
            &fs::read(directory.join(format!("reference-{base}.log"))).unwrap(),
        ),
        cache,
    );
    let expected = 24 + count * (8 + 250624 * 4);
    assert_eq!(fs::metadata(&output).unwrap().len(), expected as u64);
    output
}

fn decode_layers(bytes: &[u8], position: u32, token: u32) -> Vec<Vec<f32>> {
    assert_eq!(bytes.len(), 24 + 36 * 4096 * 4);
    assert_eq!(&bytes[..8], b"K2LAY001");
    let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(
        [word(8), word(12), word(16), word(20)],
        [36, 4096, position, token]
    );
    bytes[24..]
        .chunks_exact(4096 * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|value| {
                    let value = f32::from_le_bytes(value.try_into().unwrap());
                    assert!(value.is_finite());
                    value
                })
                .collect()
        })
        .collect()
}

#[test]
fn layer_capture_protocol_rejects_coordinate_extent_and_nonfinite_drift() {
    let mut bytes = b"K2LAY001".to_vec();
    for value in [36u32, 4096, 59, 42] {
        bytes.extend(value.to_le_bytes());
    }
    bytes.resize(24 + 36 * 4096 * 4, 0);
    assert_eq!(decode_layers(&bytes, 59, 42).len(), 36);
    for offset in [0, 8, 12, 16, 20] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(std::panic::catch_unwind(|| decode_layers(&corrupt, 59, 42)).is_err());
    }
    assert!(std::panic::catch_unwind(|| decode_layers(&bytes[..bytes.len() - 1], 59, 42)).is_err());
    bytes[24..28].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(std::panic::catch_unwind(|| decode_layers(&bytes, 59, 42)).is_err());
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
fn oracle_error_metrics_match_known_scalar_errors() {
    let metrics = error_metrics(&[1., 2.], &[4., 6.]);
    assert_eq!(metrics["max_abs"], 4.);
    assert!((metrics["rmse"].as_f64().unwrap() - 12.5_f64.sqrt()).abs() < 1e-12);
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
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(std::panic::catch_unwind(|| decode_rows(&trailing, 37, &[19], 2)).is_err());
    let mut nonfinite = bytes.clone();
    nonfinite[32..36].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert!(std::panic::catch_unwind(|| decode_rows(&nonfinite, 37, &[19], 2)).is_err());
    assert!(
        std::panic::catch_unwind(|| OracleRows::new(Cursor::new(&bytes), 37, &[19], 2).finish())
            .is_err()
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
    let f32_log = log.replace(
        ReferenceCache::F16.cache_log(),
        ReferenceCache::F32.cache_log(),
    );
    validate_reference_log_cache(&f32_log, ReferenceCache::F32);
    assert!(std::panic::catch_unwind(|| validate_reference_log(&f32_log)).is_err());
    assert!(
        std::panic::catch_unwind(|| validate_reference_log_cache(&log, ReferenceCache::F32))
            .is_err()
    );
    for (from, to) in [
        ("K (f32)", "K (f16)"),
        ("V (f32)", "V (f16)"),
        ("72.00 MiB", "36.00 MiB"),
    ] {
        assert!(
            std::panic::catch_unwind(|| validate_reference_log_cache(
                &f32_log.replace(from, to),
                ReferenceCache::F32
            ))
            .is_err()
        );
    }
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
            "native_metallib_sha256": native_kernel_identity(),
            "native_sources": native_source_identity(),
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

const PREFIX_BOUNDARIES: [usize; 12] = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256];

fn extended_inputs(tokenizer: &NativeTokenizer) -> Vec<(&'static str, u32, Vec<u32>)> {
    [
        ("ledger", 0, "The research ledger lists numbered samples, dates, locations, and measured values. Preserve every earlier entry when comparing the last sample. A red sample weighs 17 grams; a blue sample weighs 23 grams; the control weighs 19 grams. "),
        ("code", 37, "fn update(values: &mut [u32], seed: u32) -> u32 {\n    let mut sum = seed;\n    for (i, value) in values.iter_mut().enumerate() {\n        *value = value.wrapping_add(i as u32);\n        sum = sum.wrapping_add(*value);\n    }\n    sum\n}\n"),
        ("unicode", 8191, "NFC cafe\u{301}; joiners a\u{200c}b a\u{200d}b; \u{6570}\u{5b66} 123456789; \u{0645}\u{0631}\u{062d}\u{0628}\u{0627}; emoji \u{1f680}\u{1f389}; compare entries 17, 23, and 19.\n"),
    ].into_iter().map(|(name, base, text)| {
        let mut ids = tokenizer.encode(&text.repeat(32), true).unwrap().into_iter()
            .map(|id| u32::try_from(id).unwrap()).collect::<Vec<_>>();
        assert!(ids.len() >= 256);
        // A deliberately fixed teacher-forced fixture prefix, not request truncation.
        ids.truncate(256);
        (name, base, ids)
    }).collect()
}

#[test]
fn extended_prefix_boundaries_are_lengths_not_zero_based_rows() {
    assert_eq!(PREFIX_BOUNDARIES.last(), Some(&256));
    assert!(PREFIX_BOUNDARIES.windows(2).all(|pair| pair[0] < pair[1]));
    let mut previous = 0;
    let lengths = PREFIX_BOUNDARIES.map(|end| {
        let length = end - previous;
        previous = end;
        length
    });
    assert!(lengths.iter().all(|&length| length > 0));
    assert_eq!(lengths.iter().sum::<usize>(), 256);
}

#[test]
#[ignore = "GPU 256-token qualification; K2_GGUF and pinned K2_LLAMA_ORACLE; production lease; about 735 MiB evidence"]
fn gpu_256_token_corpora_match_independent_ifm_fork_and_native_splits() {
    extended_oracle_comparison(false);
}

#[test]
#[ignore = "GPU diagnostic, not qualification; same ledger IDs at three bases; production lease; about 735 MiB evidence"]
fn gpu_identical_tokens_at_three_bases_diagnostic() {
    extended_oracle_comparison(true);
}

fn extended_oracle_comparison(position_control: bool) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
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
        model_sha256,
        "5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf"
    );
    let mut cases = extended_inputs(&NativeTokenizer::from_gguf(&source).unwrap());
    if position_control {
        let ledger = cases[0].2.clone();
        cases = [0, 37, 8191]
            .into_iter()
            .map(|base| ("ledger", base, ledger.clone()))
            .collect();
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!("k2-oracle-256-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "reference":reference_identity().trim(), "reference_binary_sha256":file_sha256(&binary),
        "position_control":position_control, "qualification_attempt":!position_control,
        "model":path, "model_sha256":model_sha256, "inputs":cases, "native_capacity":256,
        "requested_reference_capacity":256, "reference_allocated_capacity":256, "cache":"F16",
        "native_q8_matvec":"lcpp", "prefix_boundaries":PREFIX_BOUNDARIES,
        "native_metallib_sha256":native_kernel_identity(),
        "native_sources":native_source_identity(),
        "thresholds":{"max_abs_exclusive":0.005,"rmse_exclusive":0.001,"cosine_exclusive_min":0.9999999,"top1_equal":true},
        "metal_api_validation":std::env::var("MTL_DEBUG_LAYER").ok(),
    })).unwrap()).unwrap();
    eprintln!("extended oracle artifacts: {}", directory.display());
    // One reference child at a time, all complete before native GPU residency.
    let outputs = cases
        .iter()
        .map(|(_, base, tokens)| run_oracle(&binary, &path, &directory, *base, tokens))
        .collect::<Vec<_>>();
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 256).unwrap();
    let mut reports = Vec::new();
    for ((name, base, tokens), output) in cases.iter().zip(outputs) {
        let mut rows = OracleRows::new(
            BufReader::new(fs::File::open(output).unwrap()),
            *base,
            tokens,
            250624,
        );
        let mut session = model.create_session(*base).unwrap();
        // Only selected native checkpoints are retained (12 MiB), never the full
        // reference sequence. Reference decoding itself holds one row at a time.
        let mut checkpoints = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            let reference = rows.row();
            let actual = session.append(&[token]).unwrap();
            let metrics = error_metrics(&actual, &reference);
            reports.push(json!({"corpus":name,"base":base,"visible_length":index + 1,
                "absolute_position":base + index as u32,"token":token,"metrics":metrics}));
            if PREFIX_BOUNDARIES.contains(&(index + 1)) {
                checkpoints.push(actual);
            }
        }
        rows.finish();
        assert_eq!(session.committed_len(), 256);
        assert!(session.append(&[42]).is_err());
        assert!(!session.is_poisoned());
        drop(session);
        let mut split = model.create_session(*base).unwrap();
        let mut start = 0;
        for (&end, expected) in PREFIX_BOUNDARIES.iter().zip(&checkpoints) {
            let actual = split.append(&tokens[start..end]).unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} split length {end}"
            );
            start = end;
        }
        drop(split);
        let mut whole = model.create_session(*base).unwrap();
        let actual = whole.append(tokens).unwrap();
        assert!(
            actual
                .iter()
                .zip(checkpoints.last().unwrap())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name} whole append"
        );
        eprintln!("{name} base={base}: 256 rows compared, split/whole bitwise controls passed");
    }
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    fs::write(
        directory.join("metrics.json"),
        serde_json::to_vec_pretty(&json!({"cases":reports})).unwrap(),
    )
    .unwrap();
    let mut failed = Vec::new();
    for report in &reports {
        let metrics = &report["metrics"];
        if metrics["actual_top1"] != metrics["reference_top1"]
            || metrics["max_abs"].as_f64().unwrap() >= 0.005
            || metrics["rmse"].as_f64().unwrap() >= 0.001
            || metrics["cosine"].as_f64().unwrap() <= 0.9999999
        {
            failed.push(report);
        }
    }
    fs::write(
        directory.join("failures.json"),
        serde_json::to_vec_pretty(&failed).unwrap(),
    )
    .unwrap();
    if position_control {
        eprintln!(
            "position-control diagnostic: {} of {} rows outside regression gates; not a qualification verdict",
            failed.len(),
            reports.len()
        );
        return;
    }
    assert!(
        failed.is_empty(),
        "{} of {} rows failed unchanged gates; inspect {}",
        failed.len(),
        reports.len(),
        directory.display()
    );
}

#[test]
#[ignore = "GPU diagnostic, not qualification; production lease; K2_GGUF and pinned K2_LLAMA_ORACLE"]
fn gpu_first_divergence_layer_diagnostic() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
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
        model_sha256,
        "5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf"
    );
    let mut cases = extended_inputs(&NativeTokenizer::from_gguf(&source).unwrap());
    for ((_, _, tokens), length) in cases.iter_mut().zip([60, 34, 27]) {
        tokens.truncate(length);
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/profiles")
        .join(format!(
            "k2-layer-diagnostic-{}-{stamp}",
            std::process::id()
        ));
    let ordinary_dir = directory.join("ordinary");
    let captured_dir = directory.join("captured");
    fs::create_dir_all(&ordinary_dir).unwrap();
    fs::create_dir(&captured_dir).unwrap();
    fs::write(directory.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "qualification":false, "purpose":"post-block residual divergence at first failing corpus rows",
        "reference":reference_identity().trim(), "reference_binary_sha256":file_sha256(&binary),
        "model":path, "model_sha256":model_sha256, "inputs":cases, "native_capacity":256,
        "cache":"F16", "native_q8_matvec":"lcpp", "capture":"last-token post-FFN residual, all 36 layers",
        "native_metallib_sha256":native_kernel_identity(),
        "native_sources":native_source_identity(),
        "metal_api_validation":std::env::var("MTL_DEBUG_LAYER").ok(),
    })).unwrap()).unwrap();
    eprintln!("layer diagnostic artifacts: {}", directory.display());
    let mut references = Vec::new();
    let mut capture_evidence = Vec::new();
    let mut attention_probes = Vec::new();
    for (_, base, tokens) in &cases {
        let ordinary = run_oracle(&binary, &path, &ordinary_dir, *base, tokens);
        let captured = run_oracle_mode(&binary, &path, &captured_dir, *base, tokens, true);
        let mut a = OracleRows::new(
            BufReader::new(fs::File::open(ordinary).unwrap()),
            *base,
            tokens,
            250624,
        );
        let mut b = OracleRows::new(
            BufReader::new(fs::File::open(&captured).unwrap()),
            *base,
            tokens,
            250624,
        );
        for _ in tokens {
            assert!(
                a.row()
                    .iter()
                    .zip(b.row())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "reference capture callback changes logits"
            );
        }
        a.finish();
        b.finish();
        let layers_path = captured.with_extension("f32.layers");
        let attention_path = captured.with_extension("f32.attention");
        capture_evidence.push(json!({"base":base,"visible_length":tokens.len(),
            "path":layers_path,"sha256":file_sha256(&layers_path),
            "attention_path":attention_path,"attention_sha256":file_sha256(&attention_path),
            "ordinary_and_captured_logits_bitwise_equal":true}));
        attention_probes.push(attention_probe::decode(
            &fs::read(attention_path).unwrap(),
            *base,
            tokens,
        ));
        references.push(decode_layers(
            &fs::read(layers_path).unwrap(),
            *base + tokens.len() as u32 - 1,
            *tokens.last().unwrap(),
        ));
    }
    fs::write(
        directory.join("captures.json"),
        serde_json::to_vec_pretty(&capture_evidence).unwrap(),
    )
    .unwrap();
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    let ctx = MetalContext::new().unwrap();
    let mut attention_reports = Vec::new();
    let config = K2HorizonConfig::from_gguf(&source).unwrap();
    for ((name, base, _), probes) in cases.iter().zip(&attention_probes) {
        for probe in probes {
            let report = attention_probe::replay(&ctx, probe, &config);
            eprintln!("{name} base={base} attention replay: {report}");
            attention_reports.push(json!({"corpus":name,"base":base,"replay":report}));
        }
    }
    fs::write(directory.join("attention-metrics.json"), serde_json::to_vec_pretty(&json!({
        "qualification":false,"reference":"F64 reductions with F32 scale, output rounded to F32",
        "cache":"captured IFM post-RoPE K and projected V rounded to F16", "cases":attention_reports
    })).unwrap()).unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 256).unwrap();
    let layers = (0..36).collect::<Vec<_>>();
    let mut reports = Vec::new();
    let mut cache_reports = Vec::new();
    for (((name, base, tokens), reference), probes) in
        cases.iter().zip(references).zip(&attention_probes)
    {
        let mut session = model.create_session(*base).unwrap();
        let actual = session.append_with_captures(tokens, &layers).unwrap();
        for probe in probes {
            let report = attention_probe::compare_cache(&session, probe);
            eprintln!("{name} base={base} cache comparison: {report}");
            cache_reports.push(json!({"corpus":name,"base":base,"cache":report}));
        }
        assert_eq!(actual.residuals.len(), 36 * 4096);
        for (layer, (a, b)) in actual
            .residuals
            .chunks_exact(4096)
            .zip(reference)
            .enumerate()
        {
            let metrics = error_metrics(a, &b);
            eprintln!("{name} layer={layer} {metrics}");
            reports.push(json!({"corpus":name,"base":base,"visible_length":tokens.len(),"layer":layer,"metrics":metrics}));
        }
    }
    fs::write(
        directory.join("cache-metrics.json"),
        serde_json::to_vec_pretty(&json!({"qualification":false,"cases":cache_reports})).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.join("metrics.json"),
        serde_json::to_vec_pretty(&json!({"qualification":false,"cases":reports})).unwrap(),
    )
    .unwrap();
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
}
