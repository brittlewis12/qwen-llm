use super::*;

const BYTES: &[u8] = include_bytes!("../../../../../../scripts/reference/k2/holdout-256-v2.json");
pub(super) const SHA256: &str = "44bd53a7dcf72ae6afb8e1f9921bf461df3f5cf9cc14c0c7705f46188418a1fe";

pub(super) fn policy() -> serde_json::Value {
    bound_policy(BYTES, SHA256)
}

#[test]
#[ignore = "GPU frozen v2 reciprocal-ranking holdout; production lease; about 5 GiB evidence; no automatic promotion"]
fn gpu_frozen_guarded_256_v2_holdout() {
    super::runner::run(policy(), SHA256, "v2", "300c8dcb", row_failures);
}

pub(super) fn row_failures(
    p: &serde_json::Value,
    metrics: &serde_json::Value,
    exact: bool,
) -> Vec<&'static str> {
    let mut failed = super::runner::numerical_failures(p, metrics);
    match ranking::classify(
        metrics,
        exact,
        p["ranking"]["two_sided_logit_regret_exclusive"]
            .as_f64()
            .unwrap(),
    ) {
        ranking::Decision::Invalid => failed.push("ranking_evidence"),
        ranking::Decision::Mismatch => failed.push("top1"),
        _ => {}
    }
    failed
}

#[test]
fn frozen_v2_keeps_numeric_controls_and_explicit_exact_trajectory_window() {
    let p = policy();
    let v1 = super::policy();
    assert_eq!(p["schema"], "k2.guarded-context-holdout.v2");
    for key in [
        "model_sha256",
        "tokenizer_metadata_id",
        "token_digest_encoding",
        "capacity",
        "bases",
        "boundaries",
        "distribution",
        "captures",
        "continuation",
    ] {
        assert_eq!(p[key], v1[key], "{key}");
    }
    let mut expected_logits = v1["logits"].clone();
    expected_logits["top1_equal"] = json!(false);
    assert_eq!(p["logits"], expected_logits);
    assert_eq!(
        p["ranking"],
        json!({"two_sided_logit_regret_exclusive":0.001,"tie_break":"highest token ID","trajectory_exact_predictor_lengths":[241,256]})
    );
    assert_eq!(
        p["corpora"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (
                c["name"].as_str().unwrap(),
                c["native_token_count"].as_u64().unwrap()
            ))
            .collect::<Vec<_>>(),
        vec![
            ("transit_budget", 352),
            ("record_decoder", 451),
            ("kitchen_translation", 375),
            ("rehearsal_schedule", 384)
        ]
    );
    let w = ranking::witness(&[1.0, 1.0005], &[1.0005, 1.0]);
    let m = json!({"logits":{"actual_top1":1,"reference_top1":0,"max_abs":0.0005,"rmse":0.0005,"cosine":0.9999999},
        "distribution":{"kl_reference_to_actual":0.0000001,"total_variation":0.0001,"centered_rmse":0.0005},"ranking":w});
    assert!(row_failures(&p, &m, false).is_empty());
    for visible in [1, 240] {
        assert!(!super::runner::exact_trajectory_predictor(&p, visible));
    }
    for visible in [241, 242, 255, 256] {
        assert!(super::runner::exact_trajectory_predictor(&p, visible));
        assert_eq!(row_failures(&p, &m, true), vec!["top1"]);
    }
    assert!(
        std::panic::catch_unwind(|| super::runner::exact_trajectory_predictor(&p, 257)).is_err()
    );
    let mut bad = m;
    bad["distribution"]["total_variation"] = json!(0.002);
    assert_eq!(row_failures(&p, &bad, false), vec!["total_variation"]);
}

#[test]
#[ignore = "CPU artifact-hash/tokenizer v2 preflight; no target forward or Metal"]
fn cpu_v2_tokenization_preflight() {
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let cases = inputs_for(&source, &policy());
    assert_eq!(cases.len(), 12);
    assert_eq!(cases.iter().filter(|c| c.3).count(), 4);
}
