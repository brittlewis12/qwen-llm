use qwen_llm::deepseek_v4_metal::{DeepSeekV4MetalResidency, DeepSeekV4PositionZeroForward};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf";
const ORACLE_BYTES: &[u8] = include_bytes!("fixtures/deepseek_v4_token35_position0_b10222.f32");
const ORACLE_MANIFEST: &str = include_str!("fixtures/deepseek_v4_token35_position0_b10222.json");

#[test]
fn pinned_position_zero_oracle_has_exact_identity() {
    let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["request"]["injected_token_id"], 35);
    assert_eq!(manifest["request"]["injection_position"], 0);
    assert_eq!(manifest["vector"]["element_count"], 129_280);
    assert_eq!(manifest["vector"]["byte_count"], ORACLE_BYTES.len());
    assert_eq!(manifest["vector"]["argmax_token_id"], 201);
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_BYTES)),
        manifest["vector"]["sha256"].as_str().unwrap()
    );
}

/// Manual only: maps the 95.93 GiB model and executes all 43 native layers.
/// This test must never be included in routine or CI test runs.
#[test]
#[ignore = "manual native DS4 full forward maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_token_35_position_zero() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );

    eprintln!("opening {}", model_path.display());
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = DeepSeekV4MetalResidency::load(&ctx, &gguf).expect("retain DS4 residency");
    eprintln!("residency={}", residency.report());
    let mut forward =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build position-zero forward");
    let started = Instant::now();
    forward
        .forward_token_zero_with_progress(&ctx, 35, |layer| {
            eprintln!(
                "completed layer {}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position-zero forward");
    let logits = forward.copy_logits_f32().expect("copy completed logits");

    assert_eq!(logits.len(), 129_280);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(ORACLE_BYTES.len(), logits.len() * 4);
    let oracle = ORACLE_BYTES
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(oracle.iter().all(|value| value.is_finite()));

    let (argmax, top_value) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("nonempty logits");
    let oracle_argmax = oracle
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap()
        .0;
    let mut dot = 0.0f64;
    let mut native_norm = 0.0f64;
    let mut oracle_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut absolute_error = 0.0f64;
    let mut maximum_error = (0usize, 0.0f32);
    for (index, (&native, &reference)) in logits.iter().zip(&oracle).enumerate() {
        let native = f64::from(native);
        let reference = f64::from(reference);
        let difference = native - reference;
        dot += native * reference;
        native_norm += native * native;
        oracle_norm += reference * reference;
        squared_error += difference * difference;
        absolute_error += difference.abs();
        if difference.abs() as f32 > maximum_error.1 {
            maximum_error = (index, difference.abs() as f32);
        }
    }
    let cosine = dot / (native_norm.sqrt() * oracle_norm.sqrt());
    let relative_rms = (squared_error / oracle_norm).sqrt();
    let mean_absolute_error = absolute_error / logits.len() as f64;
    let mut native_hasher = Sha256::new();
    for value in &logits {
        native_hasher.update(value.to_le_bytes());
    }
    let native_sha256 = format!("{:x}", native_hasher.finalize());
    eprintln!("argmax={argmax} top_logit={top_value}");
    eprintln!(
        "oracle_argmax={oracle_argmax} cosine={cosine:.9} rel_rms={relative_rms:.9} mean_abs={mean_absolute_error:.9} max_abs={} max_index={} native_sha256={native_sha256}",
        maximum_error.1, maximum_error.0
    );
    eprintln!("forward_elapsed={:.3}s", started.elapsed().as_secs_f64());

    assert_eq!(oracle_argmax, 201, "unexpected llm oracle argmax");
    assert_eq!(argmax, oracle_argmax, "native argmax differs from llm");
    assert!(cosine >= 0.999_99, "native/oracle cosine is only {cosine}");
    assert!(
        relative_rms <= 0.002,
        "native/oracle relative RMS is {relative_rms}"
    );
    assert!(
        mean_absolute_error <= 0.005,
        "native/oracle mean absolute error is {mean_absolute_error}"
    );
    assert!(
        maximum_error.1 <= 0.05,
        "native/oracle max absolute error is {:?}",
        maximum_error
    );
}
