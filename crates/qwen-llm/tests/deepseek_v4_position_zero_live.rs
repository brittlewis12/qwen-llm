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
const POSITION_126_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_pre_hca_position126_b10222.f32");
const POSITION_126_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_pre_hca_position126_b10222.json");
const POSITION_127_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x32_position127_b10222.f32");
const POSITION_127_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x32_position127_b10222.json");
const POSITION_128_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x32_then35_position128_b10222.f32");
const POSITION_128_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x32_then35_position128_b10222.json");
const POSITION_129_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x32_then35_201_position129_b10222.f32"
);
const POSITION_129_ORACLE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x32_then35_201_position129_b10222.json"
);
const POSITION_254_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_second_hca_position254_b10222.f32"
);
const POSITION_254_ORACLE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_second_hca_position254_b10222.json"
);
const POSITION_255_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x64_position255_b10222.f32");
const POSITION_255_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x64_position255_b10222.json");
const POSITION_256_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x64_then35_position256_b10222.f32");
const POSITION_256_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x64_then35_position256_b10222.json");
const SINGLETON_ORACLE_LLM_COMMIT: &str = "e07acac20fcd2ee0faca90aa91078ff142724d63";
const SINGLETON_ORACLE_LLAMA_CORE_COMMIT: &str = "b1cd3a914175adedcc976388c7c638d9a2f9a189";
const SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT: &str = "553b8e4501c57c1be08a83b6b54e549a614df162";
const SINGLETON_ORACLE_LLAMA_CPP_COMMIT: &str = "8621ac725a0b6892ae44ea33377f14b6a7e0ebdf";

struct LogitComparison {
    argmax: usize,
    oracle_argmax: usize,
    cosine: f64,
    relative_rms: f64,
    mean_absolute_error: f64,
    maximum_error: (usize, f32),
}

fn assert_hca_long_prefix_gate(label: &str, comparison: &LogitComparison) {
    assert!(
        comparison.cosine >= 0.997,
        "{label} native/oracle cosine is only {}",
        comparison.cosine
    );
    assert!(
        comparison.relative_rms <= 0.075,
        "{label} native/oracle relative RMS is {}",
        comparison.relative_rms
    );
}

fn compare_logits(label: &str, logits: &[f32], oracle_bytes: &[u8]) -> LogitComparison {
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

    LogitComparison {
        argmax,
        oracle_argmax,
        cosine,
        relative_rms,
        mean_absolute_error,
        maximum_error,
    }
}

fn assert_logits_match(label: &str, logits: &[f32], oracle_bytes: &[u8], expected_argmax: usize) {
    let comparison = compare_logits(label, logits, oracle_bytes);

    assert_eq!(
        comparison.oracle_argmax, expected_argmax,
        "unexpected {label} oracle argmax"
    );
    assert_eq!(
        comparison.argmax, comparison.oracle_argmax,
        "{label} native argmax differs from b10222"
    );
    assert!(
        comparison.cosine >= 0.999_99,
        "{label} native/oracle cosine is only {}",
        comparison.cosine
    );
    assert!(
        comparison.relative_rms <= 0.002,
        "{label} native/oracle relative RMS is {}",
        comparison.relative_rms
    );
    assert!(
        comparison.mean_absolute_error <= 0.005,
        "{label} native/oracle mean absolute error is {}",
        comparison.mean_absolute_error
    );
    assert!(
        comparison.maximum_error.1 <= 0.05,
        "{label} native/oracle max absolute error is {:?}",
        comparison.maximum_error
    );
}

#[test]
fn pinned_position_zero_oracle_has_exact_identity() {
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE_MANIFEST.as_bytes())),
        "6cbbbe3579d48290fdfaef2c67343e8c2096c55d786ede813c01bfb0cb062af9"
    );
    let manifest: serde_json::Value = serde_json::from_str(ORACLE_MANIFEST).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["request"]["injected_token_id"], 35);
    assert_eq!(manifest["request"]["injection_position"], 0);
    assert_eq!(
        manifest["request"]["prompt_decode_mode"],
        "position_zero_injection"
    );
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
        "03a7513fe075131b0602c0f795d3cccb44bcb719cf2db5dde88421e863f3bd30"
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
    assert_eq!(manifest["request"]["prompt_decode_mode"], "singleton");
    assert_eq!(
        manifest["producer"]["llm_commit"],
        SINGLETON_ORACLE_LLM_COMMIT
    );
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
        "ed3c4a3ac1f2648dcbb23a4f343c6252e80000526c5d9920b1d69c5950e2e82b"
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
    assert_eq!(manifest["request"]["prompt_decode_mode"], "singleton");
    assert_eq!(
        manifest["producer"]["llm_commit"],
        SINGLETON_ORACLE_LLM_COMMIT
    );
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
            "a141d1f7ea70a34a135eb547b648f953c6abaf40ce6cb3bc525281b6446fb8a9",
            serde_json::json!([35, 201, 200]),
            200,
            3,
            1778,
        ),
        (
            POSITION_THREE_BRANCH_ORACLE_MANIFEST,
            POSITION_THREE_BRANCH_ORACLE_BYTES,
            "2314ef2bdd85039a10fc16f6923465e391ae1d45d32ab6a435bd47fc9b57eb0a",
            serde_json::json!([35, 201, 200]),
            34,
            3,
            262,
        ),
        (
            POSITION_FOUR_BRANCH_ORACLE_MANIFEST,
            POSITION_FOUR_BRANCH_ORACLE_BYTES,
            "f3622b605aba98333228631ffa90b843cc6cbe8ca5e330f5fa8844b51e9566ab",
            serde_json::json!([35, 201, 200, 34]),
            262,
            4,
            63_325,
        ),
        (
            POSITION_SEVEN_ORACLE_MANIFEST,
            POSITION_SEVEN_ORACLE_BYTES,
            "859ddd6655099197730c77b08239bcf8ce48103677056b3c00b59bbad66c3d1d",
            serde_json::json!([35, 201, 200, 34, 35, 201, 200]),
            34,
            7,
            35,
        ),
        (
            POSITION_EIGHT_ORACLE_MANIFEST,
            POSITION_EIGHT_ORACLE_BYTES,
            "23011d7f27031e99817981aba1e27ddeb92aaa51a43d6165a6332350edc003f2",
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
        assert_eq!(
            manifest["producer"]["llm_commit"],
            SINGLETON_ORACLE_LLM_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_core_commit"],
            SINGLETON_ORACLE_LLAMA_CORE_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_cpp_rs_commit"],
            SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_cpp_commit"],
            SINGLETON_ORACLE_LLAMA_CPP_COMMIT
        );
        assert_eq!(
            manifest["model_shards_sha256"],
            serde_json::json!([
                "9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a",
                "afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd",
                "64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca",
                "5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2"
            ])
        );
        assert_eq!(manifest["request"]["prompt_decode_mode"], "singleton");
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
        assert_eq!(manifest["reproducibility"]["fresh_session_repeats"], 2);
    }
}

#[test]
fn pinned_hca_boundary_oracles_have_exact_identity() {
    let fixtures = [
        (
            POSITION_126_ORACLE_MANIFEST,
            POSITION_126_ORACLE_BYTES,
            "84d67ffb5a01afd887f7ec07f7c26437fc80890ce73d2987bbf0e8d14d5584ff",
            31,
            serde_json::json!([35, 201]),
            126,
            200,
            34,
        ),
        (
            POSITION_127_ORACLE_MANIFEST,
            POSITION_127_ORACLE_BYTES,
            "72182f2fe367afceeb5662503d92bd1ff1ebedb3ae9b1b85176af5db2d50386a",
            31,
            serde_json::json!([35, 201, 200]),
            127,
            34,
            35,
        ),
        (
            POSITION_128_ORACLE_MANIFEST,
            POSITION_128_ORACLE_BYTES,
            "4b1a0873da0e24d852cf4d9d928c6d0f99ef0ef933a09f9fe29bf0b528e8f062",
            32,
            serde_json::json!([]),
            128,
            35,
            201,
        ),
        (
            POSITION_129_ORACLE_MANIFEST,
            POSITION_129_ORACLE_BYTES,
            "df79dc5389cf1ce49b8d99f87322be02d6c1619cada993b324be2a2465512ec2",
            32,
            serde_json::json!([35]),
            129,
            201,
            200,
        ),
        (
            POSITION_254_ORACLE_MANIFEST,
            POSITION_254_ORACLE_BYTES,
            "6beced28b223259a28ffb510c1e0db01868b9fc7f3d92e6692691cdcb8b8f17e",
            63,
            serde_json::json!([35, 201]),
            254,
            200,
            34,
        ),
        (
            POSITION_255_ORACLE_MANIFEST,
            POSITION_255_ORACLE_BYTES,
            "d97b092f453c550c2e69d95ce84f77be5a79c028ec96967c15b6dfafbbf0e906",
            63,
            serde_json::json!([35, 201, 200]),
            255,
            34,
            35,
        ),
        (
            POSITION_256_ORACLE_MANIFEST,
            POSITION_256_ORACLE_BYTES,
            "4cfe6a3774910dfc90843c31f75f10812e951bcaf5ded120374a70cd858a47e0",
            64,
            serde_json::json!([]),
            256,
            35,
            201,
        ),
    ];
    for (
        manifest_source,
        vector,
        manifest_sha256,
        repetitions,
        suffix,
        position,
        injected,
        argmax,
    ) in fixtures
    {
        assert_eq!(
            format!("{:x}", Sha256::digest(manifest_source.as_bytes())),
            manifest_sha256
        );
        let manifest: serde_json::Value = serde_json::from_str(manifest_source).unwrap();
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["producer"]["effective_llama_cpp_build"], "b10222");
        assert_eq!(
            manifest["producer"]["llm_commit"],
            SINGLETON_ORACLE_LLM_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_core_commit"],
            SINGLETON_ORACLE_LLAMA_CORE_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_cpp_rs_commit"],
            SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT
        );
        assert_eq!(
            manifest["producer"]["llama_cpp_commit"],
            SINGLETON_ORACLE_LLAMA_CPP_COMMIT
        );
        assert_eq!(
            manifest["model_shards_sha256"],
            serde_json::json!([
                "9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a",
                "afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd",
                "64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca",
                "5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2"
            ])
        );
        assert_eq!(manifest["request"]["prompt_decode_mode"], "singleton");
        assert_eq!(
            manifest["request"]["prompt_tokenization"]["pattern_token_ids"],
            serde_json::json!([35, 201, 200, 34])
        );
        assert_eq!(
            manifest["request"]["prompt_tokenization"]["pattern_repetitions"],
            repetitions
        );
        assert_eq!(
            manifest["request"]["prompt_tokenization"]["suffix_token_ids"],
            suffix
        );
        assert_eq!(
            manifest["request"]["prompt_tokenization"]["expanded_token_count"],
            position
        );
        assert_eq!(
            manifest["request"]["prompt_tokenization"]["native_tokenizer_verified"],
            true
        );
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
        assert_eq!(manifest["reproducibility"]["fresh_session_repeats"], 2);
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

/// Manual only: executes two ratio-128 publications and the continuation after
/// the second row becomes visible.
#[test]
#[ignore = "manual native DS4 257-token HCA branch maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_through_second_hca_continuation() {
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
    let repeated_prefix = [35, 201, 200, 34].repeat(65);
    for &token in &repeated_prefix[..127] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 127);
    let pre_boundary_logits = session.copy_logits_f32().expect("copy position-126 logits");
    let pre_boundary = compare_logits(
        "position_126_same_session",
        &pre_boundary_logits,
        POSITION_126_ORACLE_BYTES,
    );
    assert_eq!(pre_boundary.oracle_argmax, 34);
    assert_eq!(pre_boundary.argmax, pre_boundary.oracle_argmax);
    assert!(
        pre_boundary.cosine >= 0.998,
        "position 126 native/oracle cosine is only {}",
        pre_boundary.cosine
    );
    assert!(
        pre_boundary.relative_rms <= 0.055,
        "position 126 native/oracle relative RMS is {}",
        pre_boundary.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[127], |layer| {
            eprintln!(
                "position_127 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 127");
    assert_eq!(session.next_position(), 128);
    let boundary_logits = session.copy_logits_f32().expect("copy position-127 logits");
    let boundary = compare_logits("position_127", &boundary_logits, POSITION_127_ORACLE_BYTES);
    assert_eq!(boundary.oracle_argmax, 35);
    assert_eq!(boundary.argmax, boundary.oracle_argmax);
    assert!(
        boundary.cosine >= 0.998,
        "position 127 native/oracle cosine is only {}",
        boundary.cosine
    );
    assert!(
        boundary.relative_rms <= 0.07,
        "position 127 native/oracle relative RMS is {}",
        boundary.relative_rms
    );
    assert!(
        boundary.relative_rms <= pre_boundary.relative_rms + 0.025,
        "first HCA publication introduced a discontinuity: pre={} boundary={}",
        pre_boundary.relative_rms,
        boundary.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, 35, |layer| {
            eprintln!(
                "position_128 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 128");
    assert_eq!(session.next_position(), 129);
    let continuation_logits = session.copy_logits_f32().expect("copy position-128 logits");
    let continuation = compare_logits(
        "position_128",
        &continuation_logits,
        POSITION_128_ORACLE_BYTES,
    );
    assert_eq!(continuation.oracle_argmax, 201);
    assert_eq!(continuation.argmax, continuation.oracle_argmax);
    assert!(
        continuation.cosine >= 0.998,
        "position 128 native/oracle cosine is only {}",
        continuation.cosine
    );
    assert!(
        continuation.relative_rms <= 0.06,
        "position 128 native/oracle relative RMS is {}",
        continuation.relative_rms
    );
    assert!(
        continuation.relative_rms < boundary.relative_rms,
        "HCA continuation did not recover from the boundary: boundary={} continuation={}",
        boundary.relative_rms,
        continuation.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[129], |layer| {
            eprintln!(
                "position_129 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 129");
    assert_eq!(session.next_position(), 130);
    let post_hca_logits = session.copy_logits_f32().expect("copy position-129 logits");
    let post_hca = compare_logits("position_129", &post_hca_logits, POSITION_129_ORACLE_BYTES);
    assert_eq!(post_hca.oracle_argmax, 200);
    assert_eq!(post_hca.argmax, post_hca.oracle_argmax);
    assert!(
        post_hca.cosine >= boundary.cosine - 0.002,
        "post-HCA continuation cosine regressed beyond the boundary allowance: boundary={} post={}",
        boundary.cosine,
        post_hca.cosine
    );
    assert!(
        post_hca.relative_rms <= boundary.relative_rms + 0.01,
        "post-HCA continuation relative RMS regressed beyond the boundary allowance: boundary={} post={}",
        boundary.relative_rms,
        post_hca.relative_rms
    );

    for &token in &repeated_prefix[130..254] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 254);
    session
        .forward_token_with_progress(&ctx, repeated_prefix[254], |layer| {
            eprintln!(
                "position_254 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 254");
    assert_eq!(session.next_position(), 255);
    let pre_second_hca_logits = session.copy_logits_f32().expect("copy position-254 logits");
    let pre_second_hca = compare_logits(
        "position_254",
        &pre_second_hca_logits,
        POSITION_254_ORACLE_BYTES,
    );
    assert_eq!(pre_second_hca.oracle_argmax, 34);
    assert_eq!(pre_second_hca.argmax, pre_second_hca.oracle_argmax);
    assert_hca_long_prefix_gate("position 254", &pre_second_hca);
    assert!(
        pre_second_hca.cosine >= post_hca.cosine,
        "pre-second-HCA control cosine did not recover: post={} pre_second={}",
        post_hca.cosine,
        pre_second_hca.cosine
    );
    assert!(
        pre_second_hca.relative_rms <= post_hca.relative_rms,
        "pre-second-HCA control relative RMS did not recover: post={} pre_second={}",
        post_hca.relative_rms,
        pre_second_hca.relative_rms
    );
    session
        .forward_token_with_progress(&ctx, repeated_prefix[255], |layer| {
            eprintln!(
                "position_255 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 255");
    assert_eq!(session.next_position(), 256);
    let second_hca_logits = session.copy_logits_f32().expect("copy position-255 logits");
    let second_hca = compare_logits(
        "position_255",
        &second_hca_logits,
        POSITION_255_ORACLE_BYTES,
    );
    assert_eq!(second_hca.oracle_argmax, 35);
    assert_eq!(second_hca.argmax, second_hca.oracle_argmax);
    assert_hca_long_prefix_gate("position 255", &second_hca);
    assert!(
        second_hca.cosine >= pre_second_hca.cosine,
        "second HCA publication cosine regressed from the pre-boundary control: pre={} boundary={}",
        pre_second_hca.cosine,
        second_hca.cosine
    );
    assert!(
        second_hca.relative_rms <= pre_second_hca.relative_rms,
        "second HCA publication relative RMS regressed from the pre-boundary control: pre={} boundary={}",
        pre_second_hca.relative_rms,
        second_hca.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[256], |layer| {
            eprintln!(
                "position_256 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 256");
    assert_eq!(session.next_position(), 257);
    let second_hca_continuation_logits =
        session.copy_logits_f32().expect("copy position-256 logits");
    let second_hca_continuation = compare_logits(
        "position_256",
        &second_hca_continuation_logits,
        POSITION_256_ORACLE_BYTES,
    );
    assert_eq!(second_hca_continuation.oracle_argmax, 201);
    assert_eq!(
        second_hca_continuation.argmax,
        second_hca_continuation.oracle_argmax
    );
    assert_hca_long_prefix_gate("position 256", &second_hca_continuation);
    assert!(
        second_hca_continuation.cosine >= pre_second_hca.cosine,
        "second HCA continuation cosine fell below the pre-boundary control: pre={} continuation={}",
        pre_second_hca.cosine,
        second_hca_continuation.cosine
    );
    assert!(
        second_hca_continuation.relative_rms <= pre_second_hca.relative_rms,
        "second HCA continuation relative RMS exceeded the pre-boundary control: pre={} continuation={}",
        pre_second_hca.relative_rms,
        second_hca_continuation.relative_rms
    );

    let error = session
        .forward_token(&ctx, repeated_prefix[257])
        .err()
        .expect("position 257 must reject before mutation");
    assert!(error.to_string().contains("next position is 257"));
    assert_eq!(session.next_position(), 257);
    let retained_logits = session
        .copy_logits_f32()
        .expect("position-256 logits remain completed after rejection");
    assert_eq!(retained_logits.len(), second_hca_continuation_logits.len());
    for (index, (&retained, &before)) in retained_logits
        .iter()
        .zip(&second_hca_continuation_logits)
        .enumerate()
    {
        assert_eq!(
            retained.to_bits(),
            before.to_bits(),
            "position-256 logit bits changed at index {index}"
        );
    }
    eprintln!(
        "second_hca_branch_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}
