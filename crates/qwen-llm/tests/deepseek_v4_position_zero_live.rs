use qwen_llm::deepseek_v4_metal::{DeepSeekV4MetalResidency, DeepSeekV4PositionZeroForward};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf";
const ORACLE_BYTES: &[u8] = include_bytes!("fixtures/deepseek_v4_token35_position0_b10222.f32");
const ORACLE_MANIFEST: &str = include_str!("fixtures/deepseek_v4_token35_position0_b10222.json");
const POSITION_ONE_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_position1_b10222.f32");
const POSITION_ONE_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_position1_b10222.json");
const POSITION_TWO_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_200_position2_b10222.f32");
const POSITION_TWO_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_position2_b10222.json");
const POSITION_THREE_GREEDY_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_200_200_position3_b10222.f32");
const POSITION_THREE_GREEDY_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_200_position3_b10222.json");
const POSITION_THREE_BRANCH_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_200_34_position3_b10222.f32");
const POSITION_THREE_BRANCH_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_34_position3_b10222.json");
const POSITION_FOUR_BRANCH_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_200_34_262_position4_b10222.f32");
const POSITION_FOUR_BRANCH_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_34_262_position4_b10222.json");
const POSITION_SEVEN_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_tokens35_201_200_34_35_201_200_34_position7_b10222.f32");
const POSITION_SEVEN_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_34_35_201_200_34_position7_b10222.json");
const POSITION_EIGHT_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_tokens35_201_200_34_35_201_200_34_35_position8_b10222.f32"
);
const POSITION_EIGHT_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_tokens35_201_200_34_35_201_200_34_35_position8_b10222.json");

fn assert_logits_match(label: &str, logits: &[f32], oracle_bytes: &[u8], expected_argmax: usize) {
    let oracle = oracle_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(logits.len(), oracle.len());
    assert!(logits.iter().chain(&oracle).all(|value| value.is_finite()));
    let (argmax, top_value) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
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
    for value in logits {
        native_hasher.update(value.to_le_bytes());
    }
    let native_sha256 = format!("{:x}", native_hasher.finalize());
    eprintln!("{label} argmax={argmax} top_logit={top_value}");
    eprintln!(
        "{label} oracle_argmax={oracle_argmax} cosine={cosine:.9} rel_rms={relative_rms:.9} mean_abs={mean_absolute_error:.9} max_abs={} max_index={} native_sha256={native_sha256}",
        maximum_error.1, maximum_error.0
    );

    assert_eq!(
        oracle_argmax, expected_argmax,
        "unexpected {label} oracle argmax"
    );
    assert_eq!(
        argmax, oracle_argmax,
        "{label} native argmax differs from b10222"
    );
    assert!(
        cosine >= 0.999_99,
        "{label} native/oracle cosine is only {cosine}"
    );
    assert!(
        relative_rms <= 0.002,
        "{label} native/oracle relative RMS is {relative_rms}"
    );
    assert!(
        mean_absolute_error <= 0.005,
        "{label} native/oracle mean absolute error is {mean_absolute_error}"
    );
    assert!(
        maximum_error.1 <= 0.05,
        "{label} native/oracle max absolute error is {:?}",
        maximum_error
    );
}

#[test]
fn pinned_position_zero_oracle_has_exact_identity() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_MANIFEST.as_bytes())),
        "a9106b5a2fb6266e977428ce9caf1da4bb1ea20e56b97edba22d06363a28ecff"
    );
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

#[test]
fn pinned_position_one_oracle_has_exact_identity() {
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(POSITION_ONE_ORACLE_MANIFEST.as_bytes())
        ),
        "0c762b78cad36a629c92cfb36e38bb33695790f8adcc70701ca1c8e34e5295d9"
    );
    let manifest: serde_json::Value = serde_json::from_str(POSITION_ONE_ORACLE_MANIFEST).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(
        manifest["request"]["prompt_token_ids"],
        serde_json::json!([35])
    );
    assert_eq!(manifest["request"]["injected_token_id"], 201);
    assert_eq!(manifest["request"]["injection_position"], 1);
    assert_eq!(manifest["request"]["cache_type"], "F16");
    assert_eq!(manifest["vector"]["element_count"], 129_280);
    assert_eq!(
        manifest["vector"]["byte_count"],
        POSITION_ONE_ORACLE_BYTES.len()
    );
    assert_eq!(manifest["vector"]["argmax_token_id"], 200);
    assert_eq!(
        format!("{:x}", Sha256::digest(POSITION_ONE_ORACLE_BYTES)),
        manifest["vector"]["sha256"].as_str().unwrap()
    );
}

#[test]
fn pinned_position_two_oracle_has_exact_identity() {
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(POSITION_TWO_ORACLE_MANIFEST.as_bytes())
        ),
        "d4556adba526e8a83aed0bf3502161877a391fc87d3e783f8ff327362f8569e4"
    );
    let manifest: serde_json::Value = serde_json::from_str(POSITION_TWO_ORACLE_MANIFEST).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(
        manifest["request"]["prompt_token_ids"],
        serde_json::json!([35, 201])
    );
    assert_eq!(manifest["request"]["injected_token_id"], 200);
    assert_eq!(manifest["request"]["injection_position"], 2);
    assert_eq!(manifest["request"]["cache_type"], "F16");
    assert_eq!(manifest["vector"]["element_count"], 129_280);
    assert_eq!(
        manifest["vector"]["byte_count"],
        POSITION_TWO_ORACLE_BYTES.len()
    );
    assert_eq!(manifest["vector"]["argmax_token_id"], 200);
    assert_eq!(
        format!("{:x}", Sha256::digest(POSITION_TWO_ORACLE_BYTES)),
        manifest["vector"]["sha256"].as_str().unwrap()
    );
}

#[test]
fn pinned_csa_boundary_oracles_have_exact_identity() {
    let fixtures = [
        (
            POSITION_THREE_GREEDY_ORACLE_MANIFEST,
            POSITION_THREE_GREEDY_ORACLE_BYTES,
            "b04c3838db9eac27161625a229d1d9b934583f18f90210947cd63692497ebb1d",
            serde_json::json!([35, 201, 200]),
            200,
            3,
            1778,
        ),
        (
            POSITION_THREE_BRANCH_ORACLE_MANIFEST,
            POSITION_THREE_BRANCH_ORACLE_BYTES,
            "900976d50755196ee290e9385a50d93f624ea07075625cd17bbe49b777a0a8cb",
            serde_json::json!([35, 201, 200]),
            34,
            3,
            262,
        ),
        (
            POSITION_FOUR_BRANCH_ORACLE_MANIFEST,
            POSITION_FOUR_BRANCH_ORACLE_BYTES,
            "cb2ef350438f14bd59c049f03f6e1b2fd1ec290f4c1f93734dda4d4446499ae4",
            serde_json::json!([35, 201, 200, 34]),
            262,
            4,
            63_325,
        ),
        (
            POSITION_SEVEN_ORACLE_MANIFEST,
            POSITION_SEVEN_ORACLE_BYTES,
            "462f728f0ab8d32327794f3d252eba9eff446163c2ac8bc7569919135a973e73",
            serde_json::json!([35, 201, 200, 34, 35, 201, 200]),
            34,
            7,
            35,
        ),
        (
            POSITION_EIGHT_ORACLE_MANIFEST,
            POSITION_EIGHT_ORACLE_BYTES,
            "4638324adc2c42423bbcbceb143f3db3700f8943ca9f0f644e8db9a8f0ad8b85",
            serde_json::json!([35, 201, 200, 34, 35, 201, 200, 34]),
            35,
            8,
            201,
        ),
    ];
    for (manifest_source, vector, manifest_sha256, prompt, injected, position, argmax) in fixtures {
        assert_eq!(
            format!("{:x}", Sha256::digest(manifest_source.as_bytes())),
            manifest_sha256
        );
        let manifest: serde_json::Value = serde_json::from_str(manifest_source).unwrap();
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["producer"]["effective_llama_cpp_build"], "b10222");
        assert_eq!(manifest["request"]["prompt_token_ids"], prompt);
        assert_eq!(manifest["request"]["injected_token_id"], injected);
        assert_eq!(manifest["request"]["injection_position"], position);
        assert_eq!(manifest["request"]["cache_type"], "F16");
        assert_eq!(manifest["vector"]["element_count"], 129_280);
        assert_eq!(manifest["vector"]["byte_count"], vector.len());
        assert_eq!(manifest["vector"]["argmax_token_id"], argmax);
        assert_eq!(
            format!("{:x}", Sha256::digest(vector)),
            manifest["vector"]["sha256"].as_str().unwrap()
        );
        assert_eq!(
            manifest["reproducibility"]["repeat_vectors_byte_identical"],
            true
        );
    }
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

/// Manual only: executes through the first ratio-4 compression boundary in one
/// retained native session and compares every continuing position with b10222.
#[test]
#[ignore = "manual native DS4 four-token forward maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_tokens_35_201_200_local_prefix() {
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
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build native session");
    let started = Instant::now();
    session
        .forward_token(&ctx, 35)
        .expect("execute native position zero");
    let first_logits = session
        .copy_logits_f32()
        .expect("copy position-zero logits");
    let first_argmax = first_logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_eq!(first_argmax, 201, "position-zero greedy token changed");
    eprintln!(
        "position_zero_argmax={first_argmax} elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );

    session
        .forward_token_with_progress(&ctx, 201, |layer| {
            eprintln!(
                "position_one layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position one");
    assert_eq!(session.next_position(), 2);
    let logits = session.copy_logits_f32().expect("copy position-one logits");
    let oracle = POSITION_ONE_ORACLE_BYTES
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(logits.len(), oracle.len());
    assert!(logits.iter().chain(&oracle).all(|value| value.is_finite()));

    let (argmax, top_value) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();
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
    eprintln!("two_token_elapsed={:.3}s", started.elapsed().as_secs_f64());

    assert_eq!(oracle_argmax, 200, "unexpected position-one oracle argmax");
    assert_eq!(argmax, oracle_argmax, "native argmax differs from b10222");
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

    session
        .forward_token_with_progress(&ctx, 200, |layer| {
            eprintln!(
                "position_two layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position two");
    assert_eq!(session.next_position(), 3);
    let third_logits = session.copy_logits_f32().expect("copy position-two logits");
    assert_logits_match(
        "position_two",
        &third_logits,
        POSITION_TWO_ORACLE_BYTES,
        200,
    );
    eprintln!(
        "three_token_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );

    session
        .forward_token_with_progress(&ctx, 200, |layer| {
            eprintln!(
                "position_three layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position three");
    assert_eq!(session.next_position(), 4);
    let fourth_logits = session
        .copy_logits_f32()
        .expect("copy position-three logits");
    assert_logits_match(
        "position_three_greedy",
        &fourth_logits,
        POSITION_THREE_GREEDY_ORACLE_BYTES,
        1778,
    );
    eprintln!("four_token_elapsed={:.3}s", started.elapsed().as_secs_f64());
}

/// Manual only: proves same-token CSA publication on an independently pinned
/// counterfactual branch and then advances once with the published row retained.
#[test]
#[ignore = "manual native DS4 five-token branch maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_csa_boundary_and_continuation_branch() {
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
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build native session");
    let started = Instant::now();
    for token in [35, 201, 200] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 3);

    session
        .forward_token_with_progress(&ctx, 34, |layer| {
            eprintln!(
                "branch_position_three layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute branch position three");
    assert_eq!(session.next_position(), 4);
    let boundary_logits = session
        .copy_logits_f32()
        .expect("copy branch position-three logits");
    assert_logits_match(
        "position_three_branch",
        &boundary_logits,
        POSITION_THREE_BRANCH_ORACLE_BYTES,
        262,
    );

    session
        .forward_token_with_progress(&ctx, 262, |layer| {
            eprintln!(
                "branch_position_four layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute branch position four");
    assert_eq!(session.next_position(), 5);
    let continuation_logits = session
        .copy_logits_f32()
        .expect("copy branch position-four logits");
    assert_logits_match(
        "position_four_branch",
        &continuation_logits,
        POSITION_FOUR_BRANCH_ORACLE_BYTES,
        63_325,
    );
    eprintln!(
        "five_token_branch_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}

/// Manual only: reaches the second ratio-4 boundary to prove that the first
/// overlap roll remains exact through another publication and continuation.
#[test]
#[ignore = "manual native DS4 nine-token branch maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_second_csa_boundary_and_continuation() {
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
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build native session");
    let started = Instant::now();
    for token in [35, 201, 200, 34, 35, 201, 200] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 7);

    session
        .forward_token_with_progress(&ctx, 34, |layer| {
            eprintln!(
                "position_seven layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position seven");
    assert_eq!(session.next_position(), 8);
    let boundary_logits = session
        .copy_logits_f32()
        .expect("copy position-seven logits");
    assert_logits_match(
        "position_seven",
        &boundary_logits,
        POSITION_SEVEN_ORACLE_BYTES,
        35,
    );

    session
        .forward_token_with_progress(&ctx, 35, |layer| {
            eprintln!(
                "position_eight layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position eight");
    assert_eq!(session.next_position(), 9);
    let continuation_logits = session
        .copy_logits_f32()
        .expect("copy position-eight logits");
    assert_logits_match(
        "position_eight",
        &continuation_logits,
        POSITION_EIGHT_ORACLE_BYTES,
        201,
    );
    eprintln!(
        "nine_token_branch_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}
