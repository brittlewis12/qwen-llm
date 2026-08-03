use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::deepseek_v4::DeepSeekV4Model;
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4MetalResidency, DeepSeekV4ModelContentId, DeepSeekV4PositionZeroForward,
    DeepSeekV4SnapshotCodecConstraints, DeepSeekV4SnapshotObservation, decode_causal_snapshot,
    encode_causal_snapshot, load_causal_snapshot_file, publish_causal_snapshot_file,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL: &str = "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf";
const DEFAULT_DURABLE_SNAPSHOT: &str = "target/dsv4-position1024.ds4c";
const DEFAULT_DURABLE_POSITION_2048_SNAPSHOT: &str = "target/dsv4-position2048.ds4c";
const DEFAULT_DURABLE_POSITION_2052_SNAPSHOT: &str = "target/dsv4-position2052.ds4c";
const DEFAULT_DURABLE_IDENTITY_CACHE: &str = "target/.qwen-dsv4-model-identity-v2";
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
const POSITION_257_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x64_then35_201_position257_b10222.f32"
);
const POSITION_257_ORACLE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x64_then35_201_position257_b10222.json"
);
const POSITION_382_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_third_hca_position382_b10222.f32"
);
const POSITION_382_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_pre_third_hca_position382_b10222.json");
const POSITION_383_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x96_position383_b10222.f32");
const POSITION_383_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x96_position383_b10222.json");
const POSITION_384_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x96_then35_position384_b10222.f32");
const POSITION_384_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x96_then35_position384_b10222.json");
const POSITION_384_CONTEXT_BRIDGE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x96_then35_position384_context1024_bridge_b10222.json"
);
const POSITION_385_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x96_then35_201_position385_b10222.f32"
);
const POSITION_385_ORACLE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x96_then35_201_position385_b10222.json"
);
const POSITION_510_ORACLE_BYTES: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_fourth_hca_position510_b10222.f32"
);
const POSITION_510_ORACLE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_fourth_hca_position510_b10222.json"
);
const POSITION_511_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x128_position511_b10222.f32");
const POSITION_511_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x128_position511_b10222.json");
const POSITION_512_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x128_then35_position512_b10222.f32");
const POSITION_512_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x128_then35_position512_b10222.json");
const POSITION_512_CONTEXT_2048_BRIDGE_MANIFEST: &str = include_str!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x128_then35_position512_context2048_bridge_b10222.json"
);
const POSITION_1024_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x256_then35_position1024_b10222.f32");
const POSITION_1024_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x256_then35_position1024_b10222.json");
const POSITION_2048_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x512_then35_position2048_b10222.f32");
const POSITION_2048_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x512_then35_position2048_b10222.json");
const POSITION_2051_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x513_position2051_b10222.f32");
const POSITION_2051_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x513_position2051_b10222.json");
const POSITION_2052_ORACLE_BYTES: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x513_then35_position2052_b10222.f32");
const POSITION_2052_ORACLE_MANIFEST: &str =
    include_str!("fixtures/deepseek_v4_pattern35_201_200_34_x513_then35_position2052_b10222.json");
const SINGLETON_ORACLE_LLM_COMMIT: &str = "e07acac20fcd2ee0faca90aa91078ff142724d63";
const SINGLETON_ORACLE_LLAMA_CORE_COMMIT: &str = "b1cd3a914175adedcc976388c7c638d9a2f9a189";
const SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT: &str = "553b8e4501c57c1be08a83b6b54e549a614df162";
const SINGLETON_ORACLE_LLAMA_CPP_COMMIT: &str = "8621ac725a0b6892ae44ea33377f14b6a7e0ebdf";
const MODEL_SHARDS_SHA256: [&str; 4] = [
    "9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a",
    "afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd",
    "64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca",
    "5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2",
];

struct LogitComparison {
    argmax: usize,
    oracle_argmax: usize,
    cosine: f64,
    relative_rms: f64,
    mean_absolute_error: f64,
    maximum_error: (usize, f32),
}

fn frozen_model_content_id() -> DeepSeekV4ModelContentId {
    let mut hasher = Sha256::new();
    hasher.update(b"deepseek-v4-flash-0731-ordered-shard-sha256-v1\0");
    for (index, digest) in MODEL_SHARDS_SHA256.iter().enumerate() {
        hasher.update((index as u32).to_le_bytes());
        hasher.update((digest.len() as u32).to_le_bytes());
        hasher.update(digest.as_bytes());
    }
    DeepSeekV4ModelContentId::new(hasher.finalize().into())
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

fn assert_hca_interval_drift_containment(label: &str, comparison: &LogitComparison) {
    assert!(
        comparison.cosine >= 0.990,
        "{label} native/oracle cosine is only {}",
        comparison.cosine
    );
    assert!(
        comparison.relative_rms <= 0.15,
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

fn load_admitted_residency(ctx: &MetalContext, gguf: &GgufFile) -> DeepSeekV4MetalResidency {
    let plan = DeepSeekV4MetalResidency::plan(ctx, gguf).expect("plan DS4 Metal residency");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit DS4 Metal residency");
    DeepSeekV4MetalResidency::load_from_plan(ctx, gguf, admitted)
        .expect("realize admitted DS4 Metal residency")
        .into_residency()
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
            serde_json::json!(MODEL_SHARDS_SHA256)
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
            512,
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
            512,
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
            512,
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
            512,
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
            512,
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
            512,
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
            512,
            35,
            201,
        ),
        (
            POSITION_257_ORACLE_MANIFEST,
            POSITION_257_ORACLE_BYTES,
            "81d57d211565de54303f46fea57ad13bd14dcabb562badf04aea8b85890eb237",
            64,
            serde_json::json!([35]),
            257,
            512,
            201,
            200,
        ),
        (
            POSITION_382_ORACLE_MANIFEST,
            POSITION_382_ORACLE_BYTES,
            "1fd094c0b42cfac76e29f965acfc9b2a2999d010f2d316b4b96687888c054281",
            95,
            serde_json::json!([35, 201]),
            382,
            512,
            200,
            34,
        ),
        (
            POSITION_383_ORACLE_MANIFEST,
            POSITION_383_ORACLE_BYTES,
            "16c40b9095263e2cdad926c11255ac816c6ed64827f24c81883cab7fd784f908",
            95,
            serde_json::json!([35, 201, 200]),
            383,
            512,
            34,
            35,
        ),
        (
            POSITION_384_ORACLE_MANIFEST,
            POSITION_384_ORACLE_BYTES,
            "65fd500f2f0baa412d086ed5f7842ddc114d7f4e28e1f53f8f4b182832f6e8ce",
            96,
            serde_json::json!([]),
            384,
            512,
            35,
            201,
        ),
        (
            POSITION_385_ORACLE_MANIFEST,
            POSITION_385_ORACLE_BYTES,
            "3c49fc85784e9e799bf28d3a6025abca559ef196f1c1f7d73fc051e0ea0b0178",
            96,
            serde_json::json!([35]),
            385,
            1024,
            201,
            200,
        ),
        (
            POSITION_510_ORACLE_MANIFEST,
            POSITION_510_ORACLE_BYTES,
            "cc9125e9b4b63a2075b8528b6c57f9a0e4a7053fd5d2630209b94906085ce07a",
            127,
            serde_json::json!([35, 201]),
            510,
            1024,
            200,
            34,
        ),
        (
            POSITION_511_ORACLE_MANIFEST,
            POSITION_511_ORACLE_BYTES,
            "75f59436b077eb07ed7f3edc3b52de8d3d51a2aacc086b7d9c3c6712973aa38b",
            127,
            serde_json::json!([35, 201, 200]),
            511,
            1024,
            34,
            35,
        ),
        (
            POSITION_512_ORACLE_MANIFEST,
            POSITION_512_ORACLE_BYTES,
            "f25806e7020771d47cf99f3e50a7a32688335dbf37463a740cfe1527fb40bf1d",
            128,
            serde_json::json!([]),
            512,
            1024,
            35,
            201,
        ),
        (
            POSITION_1024_ORACLE_MANIFEST,
            POSITION_1024_ORACLE_BYTES,
            "d24fc383d46dc0748ac5cb021afa55ae0be58df227ef301a7977a381a0eb5f7d",
            256,
            serde_json::json!([]),
            1024,
            2048,
            35,
            201,
        ),
        (
            POSITION_2048_ORACLE_MANIFEST,
            POSITION_2048_ORACLE_BYTES,
            "09d30384fffe423ab089894e395885e0ad5ed162a918b0e1ed650c5a30934155",
            512,
            serde_json::json!([]),
            2048,
            4096,
            35,
            201,
        ),
        (
            POSITION_2051_ORACLE_MANIFEST,
            POSITION_2051_ORACLE_BYTES,
            "54446ba2caa6a3616679de32032cab16c5d55bd1af1fbef5123684b90f30596d",
            512,
            serde_json::json!([35, 201, 200]),
            2051,
            4096,
            34,
            35,
        ),
        (
            POSITION_2052_ORACLE_MANIFEST,
            POSITION_2052_ORACLE_BYTES,
            "41e4e1b41ff5ed7846e420edc0f789bdf3eb9613657fb8be0567da71551b410c",
            513,
            serde_json::json!([]),
            2052,
            4096,
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
        context_size,
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
            serde_json::json!(MODEL_SHARDS_SHA256)
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
        assert_eq!(manifest["request"]["context_size"], context_size);
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
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(POSITION_384_CONTEXT_BRIDGE_MANIFEST.as_bytes())
        ),
        "5366d631cd24ca9762fd7320a86e5a371e0072861a475f10cf3f9c5606f52251"
    );
    let bridge_sidecar: serde_json::Value =
        serde_json::from_str(POSITION_384_CONTEXT_BRIDGE_MANIFEST).unwrap();
    assert_eq!(bridge_sidecar["purpose"], "context_capacity_bridge");
    assert_eq!(
        bridge_sidecar["producer"]["llm_commit"],
        SINGLETON_ORACLE_LLM_COMMIT
    );
    assert_eq!(
        bridge_sidecar["producer"]["llama_core_commit"],
        SINGLETON_ORACLE_LLAMA_CORE_COMMIT
    );
    assert_eq!(
        bridge_sidecar["producer"]["llama_cpp_rs_commit"],
        SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT
    );
    assert_eq!(
        bridge_sidecar["producer"]["llama_cpp_commit"],
        SINGLETON_ORACLE_LLAMA_CPP_COMMIT
    );
    assert_eq!(bridge_sidecar["request"]["context_size"], 1024);
    assert_eq!(bridge_sidecar["request"]["injection_position"], 384);
    assert_eq!(bridge_sidecar["request"]["injected_token_id"], 35);
    assert_eq!(
        format!("{:x}", Sha256::digest(POSITION_384_ORACLE_BYTES)),
        bridge_sidecar["vector"]["sha256"].as_str().unwrap()
    );
    assert_eq!(bridge_sidecar["bridge"]["source_context_size"], 512);
    assert_eq!(bridge_sidecar["bridge"]["recapture_context_size"], 1024);
    assert_eq!(bridge_sidecar["bridge"]["vectors_byte_identical"], true);

    let bridge: serde_json::Value = serde_json::from_str(POSITION_385_ORACLE_MANIFEST).unwrap();
    assert_eq!(
        bridge["reproducibility"]["context_capacity_bridge"]["position"],
        384
    );
    assert_eq!(
        bridge["reproducibility"]["context_capacity_bridge"]["context_sizes"],
        serde_json::json!([512, 1024])
    );
    assert_eq!(
        bridge["reproducibility"]["context_capacity_bridge"]["vector_sha256"],
        "310e38754af708713a92a8b6aba2be7c77f2e678b8a6f91874ede867f9d4611e"
    );
    assert_eq!(
        bridge["reproducibility"]["context_capacity_bridge"]["vectors_byte_identical"],
        true
    );

    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(POSITION_512_CONTEXT_2048_BRIDGE_MANIFEST.as_bytes())
        ),
        "8f3eb13c65c8bba6d04b6c3f36d9353d32dbf62edadbed18212afab8dcfa19be"
    );
    let long_bridge: serde_json::Value =
        serde_json::from_str(POSITION_512_CONTEXT_2048_BRIDGE_MANIFEST).unwrap();
    assert_eq!(long_bridge["purpose"], "context_capacity_bridge");
    assert_eq!(
        long_bridge["producer"]["llm_commit"],
        SINGLETON_ORACLE_LLM_COMMIT
    );
    assert_eq!(
        long_bridge["producer"]["llama_core_commit"],
        SINGLETON_ORACLE_LLAMA_CORE_COMMIT
    );
    assert_eq!(
        long_bridge["producer"]["llama_cpp_rs_commit"],
        SINGLETON_ORACLE_LLAMA_CPP_RS_COMMIT
    );
    assert_eq!(
        long_bridge["producer"]["llama_cpp_commit"],
        SINGLETON_ORACLE_LLAMA_CPP_COMMIT
    );
    assert_eq!(long_bridge["request"]["context_size"], 2048);
    assert_eq!(long_bridge["request"]["injection_position"], 512);
    assert_eq!(long_bridge["request"]["injected_token_id"], 35);
    assert_eq!(
        format!("{:x}", Sha256::digest(POSITION_512_ORACLE_BYTES)),
        long_bridge["vector"]["sha256"].as_str().unwrap()
    );
    assert_eq!(long_bridge["bridge"]["source_context_size"], 1024);
    assert_eq!(long_bridge["bridge"]["recapture_context_size"], 2048);
    assert_eq!(long_bridge["bridge"]["vectors_byte_identical"], true);

    let endpoint: serde_json::Value = serde_json::from_str(POSITION_1024_ORACLE_MANIFEST).unwrap();
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["position"],
        512
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["context_sizes"],
        serde_json::json!([1024, 2048])
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["vector_sha256"],
        "56f13995d0f9e0042a81e2015878164bd3d23a14515e093023f07edd87677376"
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["vectors_byte_identical"],
        true
    );

    let endpoint: serde_json::Value = serde_json::from_str(POSITION_2048_ORACLE_MANIFEST).unwrap();
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["position"],
        1024
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["context_sizes"],
        serde_json::json!([2048, 4096])
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["vector_sha256"],
        "6d6360c975b654d18408f262541a363a695087db4fef7722b31a5ecdf70dd03e"
    );
    assert_eq!(
        endpoint["reproducibility"]["context_capacity_bridge"]["vectors_byte_identical"],
        true
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_memory_plan_admits_and_reconciles() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let ctx = MetalContext::new().expect("create Metal context");
    let before_plan = ctx.current_allocated_size();
    let load_plan = DeepSeekV4MetalResidency::plan(&ctx, &gguf).expect("plan DS4 Metal load");
    assert_eq!(
        ctx.current_allocated_size(),
        before_plan,
        "memory planning must not realize Metal buffers"
    );
    let memory_plan = load_plan.memory_plan().clone();
    assert_eq!(memory_plan.session_allocations().len(), 539);
    assert_eq!(memory_plan.session_logical_bytes(), 179_341_856);
    assert_eq!(memory_plan.session_priced_upper_bytes(), 183_844_864);
    assert_eq!(memory_plan.residency_buffer_count(), 7);
    assert_eq!(memory_plan.residency_logical_bytes(), 102_994_608_640);
    assert_eq!(memory_plan.residency_priced_upper_bytes(), 102_994_624_512);
    assert_eq!(memory_plan.total_priced_upper_bytes(), 103_178_469_376);
    assert_eq!(
        memory_plan.required_with_reserve_bytes().unwrap(),
        103_715_340_288
    );
    assert_eq!(load_plan.residency_report().window_count, 3);
    assert_eq!(load_plan.residency_report().fallback_count, 4);
    assert_eq!(
        memory_plan.residency_logical_bytes(),
        load_plan.residency_report().resident_bytes
    );
    let signals = ctx.memory_signals();
    let admission = memory_plan.admission(signals);
    eprintln!("memory_plan={memory_plan}");
    eprintln!("memory_signals={signals:?} admission={admission:?}");
    assert!(admission.admitted, "admission denied: {admission:?}");
    assert_eq!(
        admission.reason,
        qwen_llm::metal::MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
    );

    let admitted_load_plan = load_plan.admit(signals).expect("admit DS4 load plan");
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .expect("realize admitted DS4 residency");
    let (residency, refreshed_admission, after_residency_bytes) = realized.into_parts();
    assert!(
        refreshed_admission.admitted,
        "refreshed admission denied: {refreshed_admission:?}"
    );
    assert_eq!(
        refreshed_admission.reason,
        qwen_llm::metal::MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
    );
    let before_residency_bytes = refreshed_admission.signals.current_allocated_bytes;
    memory_plan
        .reconcile_residency(before_residency_bytes, after_residency_bytes)
        .expect("reconcile DS4 residency allocation");
    let mut session = DeepSeekV4PositionZeroForward::new(&ctx, residency)
        .expect("construct admitted DS4 session");
    let after_session_bytes = ctx.current_allocated_size();
    memory_plan
        .reconcile_session(
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
        )
        .expect("reconcile DS4 session allocation");
    session
        .forward_token_zero(&ctx, 35)
        .expect("execute first admitted forward");
    let reconciliation = memory_plan
        .reconcile(qwen_llm::deepseek_v4_metal::DeepSeekV4MemorySamples {
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
            after_first_forward_bytes: ctx.current_allocated_size(),
        })
        .expect("reconcile admitted DS4 allocation samples");
    eprintln!("memory_reconciliation={reconciliation}");
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
    let residency = load_admitted_residency(&ctx, &gguf);
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

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_packed_n1_matches_position_zero_oracle() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute one-token layer-major prefill");
    assert_eq!(session.next_position(), 1);
    let logits = session.copy_logits_f32().expect("copy packed N=1 logits");
    assert_logits_match("packed position 0", &logits, ORACLE_BYTES, 201);
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_packed_n2_matches_local_continuation() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    session
        .prefill_tokens(&ctx, &[35, 201])
        .expect("execute two-token layer-major prefill");
    let packed = session.copy_logits_f32().expect("copy packed N=2 logits");
    assert_logits_match("packed position 1", &packed, POSITION_ONE_ORACLE_BYTES, 200);
    session
        .forward_token(&ctx, 200)
        .expect("decode after packed N=2");
    let continuation = session
        .copy_logits_f32()
        .expect("copy packed N=2 continuation logits");
    assert_logits_match(
        "packed N=2 continuation position 2",
        &continuation,
        POSITION_TWO_ORACLE_BYTES,
        200,
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_packed_n4_preserves_csa_and_decode_continuation() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    session
        .prefill_tokens(&ctx, &[35, 201, 200, 34])
        .expect("execute four-token layer-major prefill");
    assert_eq!(session.next_position(), 4);
    let boundary = session
        .copy_logits_f32()
        .expect("copy packed CSA-boundary logits");
    assert_logits_match(
        "packed position 3",
        &boundary,
        POSITION_THREE_BRANCH_ORACLE_BYTES,
        262,
    );

    session
        .forward_token(&ctx, 262)
        .expect("decode after packed CSA boundary");
    assert_eq!(session.next_position(), 5);
    let continuation = session
        .copy_logits_f32()
        .expect("copy packed continuation logits");
    assert_logits_match(
        "packed continuation position 4",
        &continuation,
        POSITION_FOUR_BRANCH_ORACLE_BYTES,
        63_325,
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_snapshot_restores_exact_csa_continuation() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let model_content_id = frozen_model_content_id();
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build snapshot-bound session");
    let started = Instant::now();
    session
        .prefill_tokens(&ctx, &[35, 201, 200, 34])
        .expect("execute certified CSA-boundary prefix");
    assert_eq!(session.committed_tokens(), &[35, 201, 200, 34]);
    let prefix_logits = session
        .copy_logits_f32()
        .expect("copy snapshot-prefix logits");
    assert_logits_match(
        "snapshot source position 3",
        &prefix_logits,
        POSITION_THREE_BRANCH_ORACLE_BYTES,
        262,
    );
    let prefix_snapshot = session
        .capture_causal_snapshot()
        .expect("capture position-4 causal state");
    assert_eq!(prefix_snapshot.next_position(), 4);
    assert_eq!(prefix_snapshot.prefix_tokens(), &[35, 201, 200, 34]);
    assert_eq!(
        prefix_snapshot.source_observation(),
        DeepSeekV4SnapshotObservation::Available
    );
    let snapshot_config = session.residency().config().clone();
    let constraints = DeepSeekV4SnapshotCodecConstraints {
        config: &snapshot_config,
        expected_model_content_id: model_content_id,
        max_record_bytes: 64 * 1024 * 1024,
    };
    let mut encoded_snapshot = Vec::new();
    let encoded = encode_causal_snapshot(&mut encoded_snapshot, &prefix_snapshot, constraints)
        .expect("encode causal snapshot");
    assert_eq!(encoded.record_bytes, encoded_snapshot.len() as u64);
    let durable_prefix_snapshot =
        decode_causal_snapshot(&mut Cursor::new(&encoded_snapshot), constraints)
            .expect("decode causal snapshot");
    assert_eq!(durable_prefix_snapshot, prefix_snapshot);

    session
        .forward_token(&ctx, 262)
        .expect("execute uninterrupted snapshot continuation");
    let uninterrupted_logits = session
        .copy_logits_f32()
        .expect("copy uninterrupted continuation logits");
    let uninterrupted_state = session
        .capture_causal_snapshot()
        .expect("capture uninterrupted continuation state");

    session
        .restore_causal_snapshot(&durable_prefix_snapshot)
        .expect("restore certified CSA-boundary state");
    assert_eq!(session.next_position(), 4);
    assert_eq!(session.committed_tokens(), &[35, 201, 200, 34]);
    assert!(session.copy_logits_f32().is_err());
    assert!(session.final_normalized_hidden().is_err());
    let restored_prefix = session
        .capture_causal_snapshot()
        .expect("recapture restored prefix");
    assert_eq!(
        restored_prefix.source_observation(),
        DeepSeekV4SnapshotObservation::Unavailable
    );
    assert_eq!(
        restored_prefix.causal_digest(),
        prefix_snapshot.causal_digest()
    );

    session
        .forward_token(&ctx, 262)
        .expect("execute restored snapshot continuation");
    let restored_logits = session
        .copy_logits_f32()
        .expect("copy restored continuation logits");
    assert!(
        uninterrupted_logits
            .iter()
            .zip(&restored_logits)
            .all(|(left, right)| left.to_bits() == right.to_bits()),
        "restored continuation logits must be bit-identical"
    );
    let restored_state = session
        .capture_causal_snapshot()
        .expect("capture restored continuation state");
    assert_eq!(
        restored_state.causal_digest(),
        uninterrupted_state.causal_digest()
    );
    assert_logits_match(
        "snapshot-restored position 4",
        &restored_logits,
        POSITION_FOUR_BRANCH_ORACLE_BYTES,
        63_325,
    );
    eprintln!(
        "snapshot_record_bytes={} snapshot_restore_live_elapsed={:.3}s causal_digest={}",
        encoded.record_bytes,
        started.elapsed().as_secs_f64(),
        prefix_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_packed_callback_unwind_poison_is_fail_stop() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    session
        .prefill_tokens(&ctx, &[35])
        .expect("complete a packed token with logits");
    assert!(session.copy_logits_f32().is_ok());
    assert!(session.final_normalized_hidden().is_ok());
    session
        .advance_tokens(&ctx, &[201])
        .expect("advance a retained packed token without logits");
    assert_eq!(session.next_position(), 2);
    assert!(session.copy_logits_f32().is_err());
    assert!(session.final_normalized_hidden().is_err());
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session.prefill_tokens_with_progress(&ctx, &[200, 34], |layer| {
            if layer == 2 {
                panic!("intentional packed prefill interruption");
            }
        });
    }));
    assert!(unwind.is_err(), "progress callback must interrupt prefill");
    assert_eq!(session.next_position(), 2);
    assert!(session.copy_logits_f32().is_err());
    let error = match session.forward_token(&ctx, 35) {
        Err(error) => error,
        Ok(_) => panic!("poisoned session unexpectedly accepted a token"),
    };
    assert!(error.to_string().contains("poisoned"));
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_packed_n128_preserves_hca_and_wrapped_decode() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    let prompt = [35, 201, 200, 34].repeat(32);
    let started = Instant::now();
    session
        .prefill_tokens(&ctx, &prompt)
        .expect("execute 128-token layer-major prefill");
    let packed_seconds = started.elapsed().as_secs_f64();
    eprintln!("packed_n128_seconds={packed_seconds:.3}");
    assert_eq!(session.next_position(), 128);
    let boundary_logits = session
        .copy_logits_f32()
        .expect("copy packed first-HCA logits");
    let boundary = compare_logits(
        "packed_position_127",
        &boundary_logits,
        POSITION_127_ORACLE_BYTES,
    );
    assert_eq!(boundary.oracle_argmax, 35);
    assert_eq!(boundary.argmax, boundary.oracle_argmax);
    assert!(
        boundary.cosine >= 0.998,
        "packed position 127 cosine {}",
        boundary.cosine
    );
    assert!(
        boundary.relative_rms <= 0.07,
        "packed position 127 relative RMS {}",
        boundary.relative_rms
    );

    session
        .forward_token(&ctx, 35)
        .expect("decode after packed first-HCA boundary");
    assert_eq!(session.next_position(), 129);
    let continuation_logits = session
        .copy_logits_f32()
        .expect("copy wrapped packed continuation logits");
    let continuation = compare_logits(
        "packed_position_128",
        &continuation_logits,
        POSITION_128_ORACLE_BYTES,
    );
    assert_eq!(continuation.oracle_argmax, 201);
    assert_eq!(continuation.argmax, continuation.oracle_argmax);
    assert!(
        continuation.cosine >= 0.998,
        "packed position 128 cosine {}",
        continuation.cosine
    );
    assert!(
        continuation.relative_rms <= 0.06,
        "packed position 128 relative RMS {}",
        continuation.relative_rms
    );
    assert!(
        continuation.relative_rms < boundary.relative_rms,
        "packed continuation did not recover: boundary={} continuation={}",
        boundary.relative_rms,
        continuation.relative_rms
    );
    assert!(
        packed_seconds < 17.5,
        "packed prefill {packed_seconds:.3}s did not reach 2x the 35.0s singleton baseline"
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_two_packed_chunks_match_second_hca_and_decode() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build packed session");
    let prompt = [35, 201, 200, 34].repeat(64);
    let started = Instant::now();
    session
        .advance_tokens(&ctx, &prompt[..128])
        .expect("advance first packed chunk without logits");
    assert_eq!(session.next_position(), 128);
    assert!(session.copy_logits_f32().is_err());
    session
        .prefill_tokens(&ctx, &prompt[128..])
        .expect("execute second retained packed chunk");
    assert_eq!(session.next_position(), 256);
    let boundary_logits = session
        .copy_logits_f32()
        .expect("copy second-HCA packed logits");
    let boundary = compare_logits(
        "two_chunk_position_255",
        &boundary_logits,
        POSITION_255_ORACLE_BYTES,
    );
    assert_eq!(boundary.oracle_argmax, 35);
    assert_eq!(boundary.argmax, boundary.oracle_argmax);
    assert_hca_interval_drift_containment("two-chunk position 255", &boundary);

    session
        .forward_token(&ctx, 35)
        .expect("decode after second retained packed chunk");
    assert_eq!(session.next_position(), 257);
    let continuation_logits = session
        .copy_logits_f32()
        .expect("copy two-chunk singleton continuation");
    let continuation = compare_logits(
        "two_chunk_position_256",
        &continuation_logits,
        POSITION_256_ORACLE_BYTES,
    );
    assert_eq!(continuation.oracle_argmax, 201);
    assert_eq!(continuation.argmax, continuation.oracle_argmax);
    assert_hca_long_prefix_gate("two-chunk position 256", &continuation);
    eprintln!(
        "two_packed_chunks_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "requires the local 95.93 GiB DeepSeek V4 Flash-0731 IQ3 fixture"]
fn native_deepseek_v4_retained_chunks_reach_position_1024() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    assert!(
        model_path.exists(),
        "missing DS4 model at {}",
        model_path.display()
    );
    let ctx = MetalContext::new().expect("create Metal context");
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build native session");
    let prompt = [35, 201, 200, 34].repeat(256);
    let started = Instant::now();
    for (chunk_index, chunk) in prompt.chunks(128).enumerate() {
        session
            .advance_tokens(&ctx, chunk)
            .unwrap_or_else(|error| panic!("advance packed chunk {chunk_index}: {error}"));
        assert!(session.copy_logits_f32().is_err());
        eprintln!(
            "position_1024_progress chunk={} next_position={} elapsed={:.3}s",
            chunk_index + 1,
            session.next_position(),
            started.elapsed().as_secs_f64()
        );
    }
    assert_eq!(session.next_position(), 1_024);

    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute retained packed position 1024 with logits");
    assert_eq!(session.next_position(), 1_025);
    let endpoint_logits = session
        .copy_logits_f32()
        .expect("copy position-1024 logits");
    let endpoint = compare_logits(
        "position_1024",
        &endpoint_logits,
        POSITION_1024_ORACLE_BYTES,
    );
    assert_eq!(endpoint.oracle_argmax, 201);
    assert_eq!(endpoint.argmax, endpoint.oracle_argmax);
    assert_hca_long_prefix_gate("position 1024", &endpoint);

    eprintln!(
        "position_1024_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "requires the local DS4 model and a published position-1024 snapshot"]
fn native_deepseek_v4_durable_position_1024_snapshot_matches_oracle() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let snapshot_path = std::env::var_os("DSV4_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_SNAPSHOT)
        });
    let identity_cache_path = std::env::var_os("DSV4_IDENTITY_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_IDENTITY_CACHE)
        });
    assert!(model_path.exists(), "missing DS4 model");
    assert!(snapshot_path.exists(), "missing durable DS4 snapshot");

    let started = Instant::now();
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve durable snapshot model identity");
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 config");
    let snapshot = load_causal_snapshot_file(
        &snapshot_path,
        DeepSeekV4SnapshotCodecConstraints {
            config: &model.config,
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("load durable position-1024 snapshot");
    assert_eq!(snapshot.next_position(), 1_024);
    assert_eq!(
        snapshot.prefix_tokens(),
        [35, 201, 200, 34].repeat(256).as_slice()
    );

    let ctx = MetalContext::new().expect("create Metal context");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build durable snapshot session");
    session
        .restore_causal_snapshot(&snapshot)
        .expect("restore durable position-1024 snapshot");
    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute restored position 1024");
    let logits = session
        .copy_logits_f32()
        .expect("copy durable restored endpoint logits");
    let endpoint = compare_logits("durable_position_1024", &logits, POSITION_1024_ORACLE_BYTES);
    assert_eq!(endpoint.oracle_argmax, 201);
    assert_eq!(endpoint.argmax, endpoint.oracle_argmax);
    assert_hca_long_prefix_gate("durable position 1024", &endpoint);
    let mut native_hasher = Sha256::new();
    for value in &logits {
        native_hasher.update(value.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", native_hasher.finalize()),
        "73d357295a7821607869764af42aaafc845e1764afe8c23a0aab2e5f570a7956"
    );
    eprintln!(
        "durable_position_1024_elapsed={:.3}s identity_cache={:?} causal_digest={}",
        started.elapsed().as_secs_f64(),
        content.outcome,
        snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
}

#[test]
#[ignore = "requires the local DS4 model and a published position-1024 snapshot"]
fn native_deepseek_v4_position_1024_snapshot_reaches_position_2048() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let source_snapshot_path = std::env::var_os("DSV4_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_SNAPSHOT)
        });
    let destination_snapshot_path = std::env::var_os("DSV4_POSITION_2048_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_POSITION_2048_SNAPSHOT)
        });
    let identity_cache_path = std::env::var_os("DSV4_IDENTITY_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_IDENTITY_CACHE)
        });
    assert!(model_path.exists(), "missing DS4 model");
    assert!(
        source_snapshot_path.exists(),
        "missing position-1024 snapshot"
    );

    let started = Instant::now();
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve durable snapshot model identity");
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 config");
    let source_snapshot = load_causal_snapshot_file(
        &source_snapshot_path,
        DeepSeekV4SnapshotCodecConstraints {
            config: &model.config,
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("load durable position-1024 snapshot");
    assert_eq!(source_snapshot.next_position(), 1_024);

    let ctx = MetalContext::new().expect("create Metal context");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build two-slab session");
    session
        .restore_causal_snapshot(&source_snapshot)
        .expect("restore position-1024 snapshot");
    let continuation = [35, 201, 200, 34].repeat(256);
    for (chunk_index, chunk) in continuation.chunks(128).enumerate() {
        session
            .advance_tokens(&ctx, chunk)
            .unwrap_or_else(|error| panic!("advance second-slab chunk {chunk_index}: {error}"));
        assert!(session.copy_logits_f32().is_err());
        eprintln!(
            "position_2048_progress chunk={} next_position={} elapsed={:.3}s",
            chunk_index + 1,
            session.next_position(),
            started.elapsed().as_secs_f64()
        );
    }
    assert_eq!(session.next_position(), 2_048);
    assert_eq!(session.committed_tokens(), [35, 201, 200, 34].repeat(512));

    let boundary_snapshot = session
        .capture_causal_snapshot()
        .expect("capture position-2048 snapshot");
    assert_eq!(boundary_snapshot.next_position(), 2_048);
    assert_eq!(
        boundary_snapshot.source_observation(),
        DeepSeekV4SnapshotObservation::Unavailable
    );
    assert_eq!(boundary_snapshot.payload_bytes(), 31_940_608);
    let snapshot_report = publish_causal_snapshot_file(
        &destination_snapshot_path,
        &boundary_snapshot,
        DeepSeekV4SnapshotCodecConstraints {
            config: session.residency().config(),
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("publish durable position-2048 snapshot");

    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute retained position 2048");
    assert_eq!(session.next_position(), 2_049);
    let logits = session
        .copy_logits_f32()
        .expect("copy position-2048 logits");
    let endpoint = compare_logits("position_2048", &logits, POSITION_2048_ORACLE_BYTES);
    assert_eq!(endpoint.oracle_argmax, 201);
    assert_eq!(endpoint.argmax, endpoint.oracle_argmax);
    assert_hca_long_prefix_gate("position 2048", &endpoint);
    let mut native_hasher = Sha256::new();
    for value in &logits {
        native_hasher.update(value.to_le_bytes());
    }
    let native_sha256 = format!("{:x}", native_hasher.finalize());
    assert_eq!(
        native_sha256,
        "13cb8a323341f3f23bcb98ddfb8c257b5125411065d14e1b40c5a75fa19ed53c"
    );
    assert_eq!(snapshot_report.record_bytes, 31_957_024);
    let causal_digest = boundary_snapshot
        .causal_digest()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        causal_digest,
        "a91f82475c5ecc18ae91fd607224a227c275b73745badfbf9451a63acc32e246"
    );

    let error = session
        .forward_token(&ctx, 201)
        .err()
        .expect("position 2049 must reject before mutation");
    assert!(error.to_string().contains("next position is 2049"));
    assert_eq!(session.next_position(), 2_049);
    let retained_logits = session
        .copy_logits_f32()
        .expect("position-2048 logits remain completed after rejection");
    assert!(
        retained_logits
            .iter()
            .zip(&logits)
            .all(|(retained, prior)| retained.to_bits() == prior.to_bits())
    );
    eprintln!(
        "position_2048_elapsed={:.3}s native_sha256={} snapshot_path={} snapshot_record_bytes={} causal_digest={}",
        started.elapsed().as_secs_f64(),
        native_sha256,
        destination_snapshot_path.display(),
        snapshot_report.record_bytes,
        causal_digest
    );
}

#[test]
#[ignore = "requires the local DS4 model and a published position-2048 snapshot"]
fn native_deepseek_v4_durable_position_2048_snapshot_matches_oracle() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let snapshot_path = std::env::var_os("DSV4_POSITION_2048_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_POSITION_2048_SNAPSHOT)
        });
    let identity_cache_path = std::env::var_os("DSV4_IDENTITY_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_IDENTITY_CACHE)
        });
    assert!(model_path.exists(), "missing DS4 model");
    assert!(
        snapshot_path.exists(),
        "missing durable position-2048 snapshot"
    );

    let started = Instant::now();
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve durable snapshot model identity");
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 config");
    let snapshot = load_causal_snapshot_file(
        &snapshot_path,
        DeepSeekV4SnapshotCodecConstraints {
            config: &model.config,
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("load durable position-2048 snapshot");
    assert_eq!(snapshot.next_position(), 2_048);
    assert_eq!(
        snapshot.prefix_tokens(),
        [35, 201, 200, 34].repeat(512).as_slice()
    );
    assert_eq!(snapshot.payload_bytes(), 31_940_608);
    assert_eq!(
        snapshot.source_observation(),
        DeepSeekV4SnapshotObservation::Unavailable
    );
    assert_eq!(
        snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "a91f82475c5ecc18ae91fd607224a227c275b73745badfbf9451a63acc32e246"
    );

    let ctx = MetalContext::new().expect("create Metal context");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build durable position-2048 session");
    session
        .restore_causal_snapshot(&snapshot)
        .expect("restore durable position-2048 snapshot");
    assert!(session.copy_logits_f32().is_err());
    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute restored position 2048");
    let logits = session
        .copy_logits_f32()
        .expect("copy durable position-2048 logits");
    let endpoint = compare_logits("durable_position_2048", &logits, POSITION_2048_ORACLE_BYTES);
    assert_eq!(endpoint.oracle_argmax, 201);
    assert_eq!(endpoint.argmax, endpoint.oracle_argmax);
    assert_hca_long_prefix_gate("durable position 2048", &endpoint);
    let mut native_hasher = Sha256::new();
    for value in &logits {
        native_hasher.update(value.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", native_hasher.finalize()),
        "13cb8a323341f3f23bcb98ddfb8c257b5125411065d14e1b40c5a75fa19ed53c"
    );
    eprintln!(
        "durable_position_2048_elapsed={:.3}s identity_cache={:?}",
        started.elapsed().as_secs_f64(),
        content.outcome
    );
}

#[test]
#[ignore = "requires the local DS4 model and a published position-2048 snapshot"]
fn native_deepseek_v4_position_2048_snapshot_crosses_first_sparse_csa() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let source_snapshot_path = std::env::var_os("DSV4_POSITION_2048_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_POSITION_2048_SNAPSHOT)
        });
    let destination_snapshot_path = std::env::var_os("DSV4_POSITION_2052_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_POSITION_2052_SNAPSHOT)
        });
    let identity_cache_path = std::env::var_os("DSV4_IDENTITY_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_IDENTITY_CACHE)
        });
    assert!(model_path.exists(), "missing DS4 model");
    assert!(
        source_snapshot_path.exists(),
        "missing durable position-2048 snapshot"
    );

    let started = Instant::now();
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve durable snapshot model identity");
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 config");
    let source_snapshot = load_causal_snapshot_file(
        &source_snapshot_path,
        DeepSeekV4SnapshotCodecConstraints {
            config: &model.config,
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("load durable position-2048 snapshot");
    assert_eq!(source_snapshot.next_position(), 2_048);

    let ctx = MetalContext::new().expect("create Metal context");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build sparse CSA session");
    session
        .restore_causal_snapshot(&source_snapshot)
        .expect("restore durable position-2048 snapshot");

    let mut endpoint_hashes = Vec::new();
    for (offset, token) in [35, 201, 200, 34].into_iter().enumerate() {
        let position = 2_048 + offset;
        if offset == 0 {
            session
                .prefill_tokens(&ctx, &[token])
                .unwrap_or_else(|error| panic!("execute packed position {position}: {error}"));
        } else {
            session
                .forward_token(&ctx, token)
                .unwrap_or_else(|error| panic!("execute position {position}: {error}"));
        }
        let logits = session
            .copy_logits_f32()
            .unwrap_or_else(|error| panic!("copy position {position} logits: {error}"));
        assert!(logits.iter().all(|value| value.is_finite()));
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0;
        let mut hasher = Sha256::new();
        for value in &logits {
            hasher.update(value.to_le_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());
        let (expected_argmax, expected_hash) = [
            (
                201,
                "13cb8a323341f3f23bcb98ddfb8c257b5125411065d14e1b40c5a75fa19ed53c",
            ),
            (
                200,
                "84e8d26310d87438423f3ce27c7b05f9a6567d6e72fc9d7bb0e17e589a603c25",
            ),
            (
                34,
                "e94c259038cf1272cf8ed190b1d3eeaeed2b9ae9cd93f5094cb6cd6ab080cf8d",
            ),
            (
                35,
                "de815a03d7b827d1c43859b477709b34e8ea0afd0c8eef4fed0bc2c0c8062b16",
            ),
        ][offset];
        assert_eq!(argmax, expected_argmax);
        assert_eq!(hash, expected_hash);
        if position == 2_051 {
            let boundary = compare_logits("position_2051", &logits, POSITION_2051_ORACLE_BYTES);
            assert_eq!(boundary.argmax, boundary.oracle_argmax);
            assert_hca_long_prefix_gate("position 2051", &boundary);
        }
        eprintln!(
            "sparse_position={position} token={token} argmax={argmax} sha256={hash} elapsed={:.3}s",
            started.elapsed().as_secs_f64()
        );
        endpoint_hashes.push((position, argmax, hash));
    }
    assert_eq!(session.next_position(), 2_052);

    let sparse_snapshot = session
        .capture_causal_snapshot()
        .expect("capture first sparse-CSA snapshot");
    assert_eq!(sparse_snapshot.next_position(), 2_052);
    assert_eq!(sparse_snapshot.payload_bytes(), 31_967_504);
    let report = publish_causal_snapshot_file(
        &destination_snapshot_path,
        &sparse_snapshot,
        DeepSeekV4SnapshotCodecConstraints {
            config: session.residency().config(),
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("publish position-2052 snapshot");
    assert_eq!(report.record_bytes, 31_983_920);

    session
        .forward_token(&ctx, 35)
        .expect("execute first sparse-CSA continuation at position 2052");
    let continuation_logits = session
        .copy_logits_f32()
        .expect("copy position-2052 logits");
    assert!(continuation_logits.iter().all(|value| value.is_finite()));
    let continuation_argmax = continuation_logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    let mut hasher = Sha256::new();
    for value in &continuation_logits {
        hasher.update(value.to_le_bytes());
    }
    let continuation_hash = format!("{:x}", hasher.finalize());
    assert_eq!(continuation_argmax, 201);
    assert_eq!(
        continuation_hash,
        "52e0d3bcd450bff10443ac16a93937138e3f5fce224daab6e68c76b8a0d386e6"
    );
    let continuation = compare_logits(
        "position_2052",
        &continuation_logits,
        POSITION_2052_ORACLE_BYTES,
    );
    assert_eq!(continuation.argmax, continuation.oracle_argmax);
    assert_hca_long_prefix_gate("position 2052", &continuation);
    assert_eq!(
        sparse_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "8b7906e362419dfe077eb5dd7d4909e154a5729cd100463dcfd9c2a86c172920"
    );
    eprintln!(
        "sparse_position=2052 token=35 argmax={continuation_argmax} sha256={continuation_hash} elapsed={:.3}s snapshot_digest={}",
        started.elapsed().as_secs_f64(),
        sparse_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );

    session
        .restore_causal_snapshot(&source_snapshot)
        .expect("restore position-2048 snapshot for packed sparse replay");
    session
        .prefill_tokens(&ctx, &[35, 201, 200, 34])
        .expect("execute packed chunk across first sparse-CSA boundary");
    assert_eq!(session.next_position(), 2_052);
    let packed_sparse_snapshot = session
        .capture_causal_snapshot()
        .expect("capture packed first sparse-CSA snapshot");
    session
        .prefill_tokens(&ctx, &[35])
        .expect("execute packed sparse continuation at position 2052");
    let packed_continuation_logits = session
        .copy_logits_f32()
        .expect("copy packed position-2052 logits");
    let packed_comparison = compare_logits(
        "packed_vs_singleton_position_2052",
        &packed_continuation_logits,
        bytemuck::cast_slice(&continuation_logits),
    );
    assert_eq!(packed_comparison.argmax, continuation_argmax);
    assert!(
        packed_comparison.cosine >= 0.999_99,
        "packed/singleton sparse cosine is only {}",
        packed_comparison.cosine
    );
    assert!(
        packed_comparison.relative_rms <= 0.002,
        "packed/singleton sparse relative RMS is {}",
        packed_comparison.relative_rms
    );
    let mut packed_hasher = Sha256::new();
    for value in &packed_continuation_logits {
        packed_hasher.update(value.to_le_bytes());
    }
    let packed_hash = format!("{:x}", packed_hasher.finalize());
    assert_eq!(
        packed_hash,
        "d78f88e7acd4746477c25bc899eefa78af97a42cba266141ec7c76ab27e96b9f"
    );
    assert_eq!(
        packed_sparse_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "03cea8187af6782fa374bf8ce441057d74924880eb6be46439b25982098476db"
    );
    eprintln!(
        "packed_sparse_position=2052 sha256={} snapshot_digest={} singleton_snapshot_digest={} elapsed={:.3}s",
        packed_hash,
        packed_sparse_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        sparse_snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        started.elapsed().as_secs_f64()
    );

    let error = session
        .forward_token(&ctx, continuation_argmax as u32)
        .err()
        .expect("position 2053 must reject before mutation");
    assert!(error.to_string().contains("next position is 2053"));
    assert_eq!(session.next_position(), 2_053);
    assert_eq!(endpoint_hashes.len(), 4);
}

#[test]
#[ignore = "requires the local DS4 model and a published position-2052 snapshot"]
fn native_deepseek_v4_durable_position_2052_snapshot_matches_oracle() {
    let model_path = std::env::var_os("DSV4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let snapshot_path = std::env::var_os("DSV4_POSITION_2052_SNAPSHOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_POSITION_2052_SNAPSHOT)
        });
    let identity_cache_path = std::env::var_os("DSV4_IDENTITY_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(DEFAULT_DURABLE_IDENTITY_CACHE)
        });
    assert!(model_path.exists(), "missing DS4 model");
    assert!(
        snapshot_path.exists(),
        "missing durable position-2052 snapshot"
    );

    let started = Instant::now();
    let gguf = GgufFile::open(&model_path).expect("open DS4 GGUF shards");
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(identity_cache_path))
            .expect("resolve durable sparse snapshot model identity");
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 config");
    let snapshot = load_causal_snapshot_file(
        &snapshot_path,
        DeepSeekV4SnapshotCodecConstraints {
            config: &model.config,
            expected_model_content_id: model_content_id,
            max_record_bytes: 64 * 1024 * 1024,
        },
    )
    .expect("load durable position-2052 snapshot");
    assert_eq!(snapshot.next_position(), 2_052);
    assert_eq!(
        snapshot.prefix_tokens(),
        [35, 201, 200, 34].repeat(513).as_slice()
    );
    assert_eq!(snapshot.payload_bytes(), 31_967_504);
    assert_eq!(
        snapshot.source_observation(),
        DeepSeekV4SnapshotObservation::Available
    );
    assert_eq!(
        snapshot
            .causal_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "8b7906e362419dfe077eb5dd7d4909e154a5729cd100463dcfd9c2a86c172920"
    );

    let ctx = MetalContext::new().expect("create Metal context");
    let residency = load_admitted_residency(&ctx, &gguf);
    let mut session =
        DeepSeekV4PositionZeroForward::new_with_model_content_id(&ctx, residency, model_content_id)
            .expect("build durable position-2052 session");
    session
        .restore_causal_snapshot(&snapshot)
        .expect("restore durable position-2052 snapshot");
    assert!(session.copy_logits_f32().is_err());
    session
        .forward_token(&ctx, 35)
        .expect("execute restored position 2052");
    let logits = session
        .copy_logits_f32()
        .expect("copy durable position-2052 logits");
    let endpoint = compare_logits("durable_position_2052", &logits, POSITION_2052_ORACLE_BYTES);
    assert_eq!(endpoint.oracle_argmax, 201);
    assert_eq!(endpoint.argmax, endpoint.oracle_argmax);
    assert_hca_long_prefix_gate("durable position 2052", &endpoint);
    let mut native_hasher = Sha256::new();
    for value in &logits {
        native_hasher.update(value.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", native_hasher.finalize()),
        "52e0d3bcd450bff10443ac16a93937138e3f5fce224daab6e68c76b8a0d386e6"
    );
    eprintln!(
        "durable_position_2052_elapsed={:.3}s identity_cache={:?}",
        started.elapsed().as_secs_f64(),
        content.outcome
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
    let residency = load_admitted_residency(&ctx, &gguf);
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
    let residency = load_admitted_residency(&ctx, &gguf);
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
    let residency = load_admitted_residency(&ctx, &gguf);
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

/// Manual only: executes four ratio-128 publications and the continuation
/// after the fourth row becomes visible.
#[test]
#[ignore = "manual native DS4 513-token HCA branch maps the 95.93 GiB checkpoint"]
fn native_deepseek_v4_through_fourth_hca_continuation() {
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
    let residency = load_admitted_residency(&ctx, &gguf);
    eprintln!("residency={}", residency.report());
    let mut session =
        DeepSeekV4PositionZeroForward::new(&ctx, residency).expect("build native session");
    let started = Instant::now();
    let repeated_prefix = [35, 201, 200, 34].repeat(129);
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

    session
        .forward_token_with_progress(&ctx, repeated_prefix[257], |layer| {
            eprintln!(
                "position_257 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 257");
    assert_eq!(session.next_position(), 258);
    let interval_start_logits = session.copy_logits_f32().expect("copy position-257 logits");
    let interval_start = compare_logits(
        "position_257",
        &interval_start_logits,
        POSITION_257_ORACLE_BYTES,
    );
    assert_eq!(interval_start.oracle_argmax, 200);
    assert_eq!(interval_start.argmax, interval_start.oracle_argmax);
    assert_hca_long_prefix_gate("position 257", &interval_start);

    for &token in &repeated_prefix[258..382] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 382);
    session
        .forward_token_with_progress(&ctx, repeated_prefix[382], |layer| {
            eprintln!(
                "position_382 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 382");
    assert_eq!(session.next_position(), 383);
    let pre_third_hca_logits = session.copy_logits_f32().expect("copy position-382 logits");
    let pre_third_hca = compare_logits(
        "position_382",
        &pre_third_hca_logits,
        POSITION_382_ORACLE_BYTES,
    );
    assert_eq!(pre_third_hca.oracle_argmax, 34);
    assert_eq!(pre_third_hca.argmax, pre_third_hca.oracle_argmax);
    assert_hca_interval_drift_containment("position 382", &pre_third_hca);

    session
        .forward_token_with_progress(&ctx, repeated_prefix[383], |layer| {
            eprintln!(
                "position_383 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 383");
    assert_eq!(session.next_position(), 384);
    let third_hca_logits = session.copy_logits_f32().expect("copy position-383 logits");
    let third_hca = compare_logits("position_383", &third_hca_logits, POSITION_383_ORACLE_BYTES);
    assert_eq!(third_hca.oracle_argmax, 35);
    assert_eq!(third_hca.argmax, third_hca.oracle_argmax);
    assert_hca_interval_drift_containment("position 383", &third_hca);
    assert!(
        third_hca.cosine >= pre_third_hca.cosine,
        "third HCA publication cosine regressed from the pre-boundary control: pre={} boundary={}",
        pre_third_hca.cosine,
        third_hca.cosine
    );
    assert!(
        third_hca.relative_rms <= pre_third_hca.relative_rms,
        "third HCA publication relative RMS regressed from the pre-boundary control: pre={} boundary={}",
        pre_third_hca.relative_rms,
        third_hca.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[384], |layer| {
            eprintln!(
                "position_384 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 384");
    assert_eq!(session.next_position(), 385);
    let third_hca_continuation_logits =
        session.copy_logits_f32().expect("copy position-384 logits");
    let third_hca_continuation = compare_logits(
        "position_384",
        &third_hca_continuation_logits,
        POSITION_384_ORACLE_BYTES,
    );
    assert_eq!(third_hca_continuation.oracle_argmax, 201);
    assert_eq!(
        third_hca_continuation.argmax,
        third_hca_continuation.oracle_argmax
    );
    assert_hca_long_prefix_gate("position 384", &third_hca_continuation);
    assert!(
        third_hca_continuation.cosine > third_hca.cosine,
        "third HCA continuation cosine did not recover: boundary={} continuation={}",
        third_hca.cosine,
        third_hca_continuation.cosine
    );
    assert!(
        third_hca_continuation.relative_rms < third_hca.relative_rms,
        "third HCA continuation relative RMS did not recover: boundary={} continuation={}",
        third_hca.relative_rms,
        third_hca_continuation.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[385], |layer| {
            eprintln!(
                "position_385 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 385");
    assert_eq!(session.next_position(), 386);
    let fourth_interval_start_logits = session.copy_logits_f32().expect("copy position-385 logits");
    let fourth_interval_start = compare_logits(
        "position_385",
        &fourth_interval_start_logits,
        POSITION_385_ORACLE_BYTES,
    );
    assert_eq!(fourth_interval_start.oracle_argmax, 200);
    assert_eq!(
        fourth_interval_start.argmax,
        fourth_interval_start.oracle_argmax
    );
    assert_hca_interval_drift_containment("position 385", &fourth_interval_start);

    for &token in &repeated_prefix[386..510] {
        session.forward_token(&ctx, token).unwrap();
    }
    assert_eq!(session.next_position(), 510);
    session
        .forward_token_with_progress(&ctx, repeated_prefix[510], |layer| {
            eprintln!(
                "position_510 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 510");
    assert_eq!(session.next_position(), 511);
    let pre_fourth_hca_logits = session.copy_logits_f32().expect("copy position-510 logits");
    let pre_fourth_hca = compare_logits(
        "position_510",
        &pre_fourth_hca_logits,
        POSITION_510_ORACLE_BYTES,
    );
    assert_eq!(pre_fourth_hca.oracle_argmax, 34);
    assert_eq!(pre_fourth_hca.argmax, pre_fourth_hca.oracle_argmax);
    assert_hca_interval_drift_containment("position 510", &pre_fourth_hca);

    session
        .forward_token_with_progress(&ctx, repeated_prefix[511], |layer| {
            eprintln!(
                "position_511 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 511");
    assert_eq!(session.next_position(), 512);
    let fourth_hca_logits = session.copy_logits_f32().expect("copy position-511 logits");
    let fourth_hca = compare_logits(
        "position_511",
        &fourth_hca_logits,
        POSITION_511_ORACLE_BYTES,
    );
    assert_eq!(fourth_hca.oracle_argmax, 35);
    assert_eq!(fourth_hca.argmax, fourth_hca.oracle_argmax);
    assert_hca_interval_drift_containment("position 511", &fourth_hca);
    assert!(
        fourth_hca.cosine >= pre_fourth_hca.cosine,
        "fourth HCA publication cosine regressed from the pre-boundary control: pre={} boundary={}",
        pre_fourth_hca.cosine,
        fourth_hca.cosine
    );
    assert!(
        fourth_hca.relative_rms <= pre_fourth_hca.relative_rms,
        "fourth HCA publication relative RMS regressed from the pre-boundary control: pre={} boundary={}",
        pre_fourth_hca.relative_rms,
        fourth_hca.relative_rms
    );

    session
        .forward_token_with_progress(&ctx, repeated_prefix[512], |layer| {
            eprintln!(
                "position_512 layer={}/43 elapsed={:.3}s",
                layer + 1,
                started.elapsed().as_secs_f64()
            );
        })
        .expect("execute native position 512");
    assert_eq!(session.next_position(), 513);
    let fourth_hca_continuation_logits =
        session.copy_logits_f32().expect("copy position-512 logits");
    let fourth_hca_continuation = compare_logits(
        "position_512",
        &fourth_hca_continuation_logits,
        POSITION_512_ORACLE_BYTES,
    );
    assert_eq!(fourth_hca_continuation.oracle_argmax, 201);
    assert_eq!(
        fourth_hca_continuation.argmax,
        fourth_hca_continuation.oracle_argmax
    );
    assert_hca_long_prefix_gate("position 512", &fourth_hca_continuation);
    assert!(
        fourth_hca_continuation.cosine > fourth_hca.cosine,
        "fourth HCA continuation cosine did not recover: boundary={} continuation={}",
        fourth_hca.cosine,
        fourth_hca_continuation.cosine
    );
    assert!(
        fourth_hca_continuation.relative_rms < fourth_hca.relative_rms,
        "fourth HCA continuation relative RMS did not recover: boundary={} continuation={}",
        fourth_hca.relative_rms,
        fourth_hca_continuation.relative_rms
    );

    let error = session
        .forward_token(&ctx, repeated_prefix[513])
        .err()
        .expect("position 513 must reject before mutation");
    assert!(error.to_string().contains("next position is 513"));
    assert_eq!(session.next_position(), 513);
    let retained_logits = session
        .copy_logits_f32()
        .expect("position-512 logits remain completed after rejection");
    assert_eq!(retained_logits.len(), fourth_hca_continuation_logits.len());
    for (index, (&retained, &before)) in retained_logits
        .iter()
        .zip(&fourth_hca_continuation_logits)
        .enumerate()
    {
        assert_eq!(
            retained.to_bits(),
            before.to_bits(),
            "position-512 logit bits changed at index {index}"
        );
    }
    eprintln!(
        "fourth_hca_branch_elapsed={:.3}s",
        started.elapsed().as_secs_f64()
    );
}
