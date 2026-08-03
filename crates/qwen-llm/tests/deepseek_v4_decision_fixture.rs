use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const NATIVE_SOURCE: &str = include_str!("fixtures/deepseek_v4_position3070_native_decisions.json");
const BATCHED_SOURCE: &str =
    include_str!("fixtures/deepseek_v4_position3070_b10222_batched_decisions.json");
const SINGLETON_SOURCE: &str =
    include_str!("fixtures/deepseek_v4_position3070_b10222_singleton_decisions.json");
const MANIFEST_SOURCE: &str =
    include_str!("fixtures/deepseek_v4_position3070_3072_schedule_envelope_b10222.json");
const LLM_PATCH: &[u8] =
    include_bytes!("fixtures/deepseek_v4_b10222_decision_transcript_llm.patch");
const LLAMA_CORE_PATCH: &[u8] =
    include_bytes!("fixtures/deepseek_v4_b10222_decision_transcript_llama_core.patch");
const POSITION_3070_SINGLETON: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_twenty_fourth_hca_position3070_b10222.f32"
);
const POSITION_3070_BATCHED: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_pre_twenty_fourth_hca_position3070_b10222_batched.f32"
);
const POSITION_3071_SINGLETON: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x768_position3071_b10222.f32");
const POSITION_3071_BATCHED: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x768_position3071_b10222_batched.f32"
);
const POSITION_3072_SINGLETON: &[u8] =
    include_bytes!("fixtures/deepseek_v4_pattern35_201_200_34_x768_then35_position3072_b10222.f32");
const POSITION_3072_BATCHED: &[u8] = include_bytes!(
    "fixtures/deepseek_v4_pattern35_201_200_34_x768_then35_position3072_b10222_batched.f32"
);
const CSA_LAYERS: [usize; 21] = [
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
];

#[derive(Clone, Debug)]
struct CsaDecision {
    scores: Vec<f32>,
    selected: Vec<u32>,
}

#[derive(Clone, Debug)]
struct LayerDecision {
    csa: Option<CsaDecision>,
    route_ids: Vec<u32>,
    route_weights: Vec<f32>,
}

fn json_f32(values: &Value) -> Vec<f32> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect()
}

fn json_u32(values: &Value) -> Vec<u32> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_i64().unwrap()).unwrap())
        .collect()
}

fn parse_native(source: &str) -> BTreeMap<usize, LayerDecision> {
    let root: Value = serde_json::from_str(source).unwrap();
    assert_eq!(root["schema_version"], 1);
    assert_eq!(root["position"], 3_070);
    assert_eq!(root["layer_count"], 43);
    assert_eq!(root["csa_layer_count"], 21);
    let layers = root["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 43);
    layers
        .iter()
        .map(|layer| {
            let index = layer["layer"].as_u64().unwrap() as usize;
            let csa = layer["csa"].as_object().map(|csa| CsaDecision {
                scores: json_f32(&csa["visible_scores"]),
                selected: json_u32(&csa["cache_order_selected_ids"]),
            });
            let route = &layer["route"];
            (
                index,
                LayerDecision {
                    csa,
                    route_ids: json_u32(&route["expert_ids"]),
                    route_weights: json_f32(&route["normalized_scaled_weights"]),
                },
            )
        })
        .collect()
}

fn parse_oracle(source: &str, expected_mode: &str) -> BTreeMap<usize, LayerDecision> {
    let root: Value = serde_json::from_str(source).unwrap();
    assert_eq!(
        root["schema"],
        "deepseek_v4_injected_decision_transcript/v1"
    );
    assert_eq!(root["position"], 3_070);
    assert_eq!(root["token_id"], 200);
    assert_eq!(root["capture_errors"], serde_json::json!([]));
    assert_eq!(root["producer"]["decode_tokens"], 1);
    assert_eq!(root["producer"]["batch_size"], 512);
    assert_eq!(root["producer"]["ubatch_size"], 128);
    assert_eq!(
        root["snapshot_id"],
        match expected_mode {
            "batched" => "dsv4_position3070_ctx4096_token200_batched_trace",
            "singleton" => "dsv4_position3070_ctx4096_token200_singleton_trace",
            mode => panic!("unexpected oracle mode {mode}"),
        }
    );

    let tensors = root["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| (tensor["name"].as_str().unwrap(), tensor))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(tensors.len(), 128);
    (0..43)
        .map(|layer| {
            let route_ids = tensors[format!("ffn_moe_topk-{layer}").as_str()];
            let route_weights = tensors[format!("ffn_moe_weights_scaled-{layer}").as_str()];
            assert_eq!(route_ids["dtype"], "i32");
            assert_eq!(route_ids["shape"], serde_json::json!([6, 1, 1, 1]));
            assert_eq!(route_weights["dtype"], "f32");
            assert_eq!(route_weights["shape"], serde_json::json!([1, 6, 1, 1]));
            let csa = CSA_LAYERS.contains(&layer).then(|| {
                let scores = tensors[format!("lid_score_masked-{layer}").as_str()];
                let selected = tensors[format!("lid_top_k-{layer}").as_str()];
                assert_eq!(scores["dtype"], "f32");
                assert_eq!(scores["shape"], serde_json::json!([768, 1, 1, 1]));
                assert_eq!(selected["dtype"], "i32");
                assert_eq!(selected["shape"], serde_json::json!([512, 1, 1, 1]));
                let scores = scores["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .take(767)
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                let mut selected = json_u32(&selected["values"]);
                selected.sort_unstable();
                CsaDecision { scores, selected }
            });
            (
                layer,
                LayerDecision {
                    csa,
                    route_ids: json_u32(&route_ids["values"]),
                    route_weights: json_f32(&route_weights["values"]),
                },
            )
        })
        .collect()
}

fn stable_top_512(scores: &[f32]) -> Vec<u32> {
    assert_eq!(scores.len(), 767);
    assert!(scores.iter().all(|score| score.is_finite()));
    let mut ranked = (0..scores.len()).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });
    let mut selected = ranked[..512]
        .iter()
        .map(|&row| row as u32)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

fn selected_set(layer: &LayerDecision) -> BTreeSet<u32> {
    layer
        .csa
        .as_ref()
        .unwrap()
        .selected
        .iter()
        .copied()
        .collect()
}

fn set_difference(left: &BTreeSet<u32>, right: &BTreeSet<u32>) -> Vec<u32> {
    left.difference(right).copied().collect()
}

fn first_csa_set_difference(
    left: &BTreeMap<usize, LayerDecision>,
    right: &BTreeMap<usize, LayerDecision>,
) -> Option<usize> {
    CSA_LAYERS
        .into_iter()
        .find(|layer| selected_set(&left[layer]) != selected_set(&right[layer]))
}

fn first_route_difference(
    left: &BTreeMap<usize, LayerDecision>,
    right: &BTreeMap<usize, LayerDecision>,
    compare_as_set: bool,
) -> Option<usize> {
    (0..43).find(|layer| {
        if compare_as_set {
            left[layer]
                .route_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != right[layer]
                    .route_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
        } else {
            left[layer].route_ids != right[layer].route_ids
        }
    })
}

fn assert_row_order(scores: &[f32], selected: &[u32], rejected: &[u32]) {
    let selected_floor = selected
        .iter()
        .map(|&row| scores[row as usize])
        .fold(f32::INFINITY, f32::min);
    let rejected_ceiling = rejected
        .iter()
        .map(|&row| scores[row as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(selected_floor > rejected_ceiling);
}

fn score_metrics(left: &[f32], right: &[f32]) -> (f64, f64) {
    assert_eq!(left.len(), right.len());
    assert!(left.iter().chain(right).all(|score| score.is_finite()));
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    let mut squared_error = 0.0_f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
        squared_error += (left - right).powi(2);
    }
    (
        dot / (left_norm.sqrt() * right_norm.sqrt()),
        (squared_error / right_norm).sqrt(),
    )
}

fn cutoff_margin(scores: &[f32]) -> f64 {
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    f64::from(ranked[511].1 - ranked[512].1)
}

fn assert_near(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected:.12}, got {actual:.12}"
    );
}

fn assert_cutoff_bifurcation(
    label: &str,
    left: &[f32],
    right: &[f32],
    reversed_rows: &[u32],
    expected: &Value,
) {
    let (cosine, relative_rms) = score_metrics(left, right);
    let left_margin = cutoff_margin(left);
    let right_margin = cutoff_margin(right);
    let minimum_reversed_row_perturbation = reversed_rows
        .iter()
        .map(|&row| f64::from((left[row as usize] - right[row as usize]).abs()))
        .fold(f64::INFINITY, f64::min);
    assert_near(
        &format!("{label} score cosine"),
        cosine,
        expected["score_cosine"].as_f64().unwrap(),
        5e-9,
    );
    assert_near(
        &format!("{label} score relative RMS"),
        relative_rms,
        expected["score_relative_rms"].as_f64().unwrap(),
        5e-8,
    );
    assert_near(
        &format!("{label} left cutoff margin"),
        left_margin,
        expected["left_cutoff_margin"].as_f64().unwrap(),
        2e-7,
    );
    assert_near(
        &format!("{label} right cutoff margin"),
        right_margin,
        expected["right_cutoff_margin"].as_f64().unwrap(),
        2e-7,
    );
    assert_near(
        &format!("{label} minimum reversed-row perturbation"),
        minimum_reversed_row_perturbation,
        expected["minimum_reversed_row_perturbation"]
            .as_f64()
            .unwrap(),
        2e-7,
    );
    assert_eq!(expected["perturbation_exceeds_each_cutoff_margin"], true);
    assert!(minimum_reversed_row_perturbation > left_margin);
    assert!(minimum_reversed_row_perturbation > right_margin);
}

#[test]
fn pinned_position_3070_decisions_explain_long_sparse_schedule_drift() {
    assert_eq!(
        format!("{:x}", Sha256::digest(NATIVE_SOURCE.as_bytes())),
        "539c0cfac45ff2ead361a5f3c0d9e2730347810c41fac5b3fdf006a4360a7c46"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(BATCHED_SOURCE.as_bytes())),
        "771f3c5333c928aba1a7211d1880677bc62ef1417c22b24c1060b24e849ec827"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(SINGLETON_SOURCE.as_bytes())),
        "7ca6a3f65506c7ee3ece13b7556a2b12277356ff303283ee95da27a9d842df4a"
    );

    let manifest: Value = serde_json::from_str(MANIFEST_SOURCE).unwrap();
    let score_bifurcations = &manifest["mechanism_gate"]["score_bifurcations"];
    let native = parse_native(NATIVE_SOURCE);
    let batched = parse_oracle(BATCHED_SOURCE, "batched");
    let singleton = parse_oracle(SINGLETON_SOURCE, "singleton");
    for transcript in [&native, &batched, &singleton] {
        assert_eq!(transcript.len(), 43);
        for layer in 0..43 {
            assert_eq!(transcript[&layer].route_ids.len(), 6);
            assert_eq!(transcript[&layer].route_weights.len(), 6);
            assert!(
                transcript[&layer]
                    .route_weights
                    .iter()
                    .all(|weight| weight.is_finite())
            );
            assert_eq!(
                transcript[&layer].csa.is_some(),
                CSA_LAYERS.contains(&layer)
            );
            if let Some(csa) = &transcript[&layer].csa {
                assert_eq!(csa.selected.len(), 512);
                assert_eq!(csa.selected, stable_top_512(&csa.scores));
            }
        }
    }

    let layer_two = selected_set(&native[&2]);
    assert_eq!(layer_two, selected_set(&batched[&2]));
    assert_eq!(layer_two, selected_set(&singleton[&2]));

    assert_eq!(first_csa_set_difference(&native, &batched), Some(4));
    assert_eq!(first_csa_set_difference(&native, &singleton), Some(4));
    assert_eq!(first_csa_set_difference(&batched, &singleton), Some(6));
    let native_four = selected_set(&native[&4]);
    let batched_four = selected_set(&batched[&4]);
    let singleton_four = selected_set(&singleton[&4]);
    assert_eq!(batched_four, singleton_four);
    assert_eq!(set_difference(&native_four, &batched_four), [378, 381]);
    assert_eq!(set_difference(&batched_four, &native_four), [383, 384]);
    assert_row_order(
        &native[&4].csa.as_ref().unwrap().scores,
        &[378, 381],
        &[383, 384],
    );
    for oracle in [&batched, &singleton] {
        assert_row_order(
            &oracle[&4].csa.as_ref().unwrap().scores,
            &[383, 384],
            &[378, 381],
        );
    }
    assert_cutoff_bifurcation(
        "native/batched layer 4",
        &native[&4].csa.as_ref().unwrap().scores,
        &batched[&4].csa.as_ref().unwrap().scores,
        &[378, 381, 383, 384],
        &score_bifurcations["native_vs_batched_layer4"],
    );
    assert_cutoff_bifurcation(
        "native/singleton layer 4",
        &native[&4].csa.as_ref().unwrap().scores,
        &singleton[&4].csa.as_ref().unwrap().scores,
        &[378, 381, 383, 384],
        &score_bifurcations["native_vs_singleton_layer4"],
    );

    let batched_six = selected_set(&batched[&6]);
    let singleton_six = selected_set(&singleton[&6]);
    assert_eq!(set_difference(&batched_six, &singleton_six), [386]);
    assert_eq!(set_difference(&singleton_six, &batched_six), [295]);
    assert_row_order(&batched[&6].csa.as_ref().unwrap().scores, &[386], &[295]);
    assert_row_order(&singleton[&6].csa.as_ref().unwrap().scores, &[295], &[386]);
    assert_cutoff_bifurcation(
        "batched/singleton layer 6",
        &batched[&6].csa.as_ref().unwrap().scores,
        &singleton[&6].csa.as_ref().unwrap().scores,
        &[386, 295],
        &score_bifurcations["batched_vs_singleton_layer6"],
    );

    for (left, right) in [
        (&native, &batched),
        (&native, &singleton),
        (&batched, &singleton),
    ] {
        assert_eq!(first_route_difference(left, right, false), Some(5));
        assert_eq!(first_route_difference(left, right, true), Some(7));
    }
}

#[test]
fn schedule_envelope_manifest_pins_producer_vectors_and_repeats() {
    assert_eq!(
        format!("{:x}", Sha256::digest(MANIFEST_SOURCE.as_bytes())),
        "ba26fa36e8ba6c0f776c3c2783cf23c305c60f9b1d06298bab41c8347d80c067"
    );
    let manifest: Value = serde_json::from_str(MANIFEST_SOURCE).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["producer"]["effective_llama_cpp_build"], "b10222");
    assert_eq!(
        manifest["producer"]["llm_commit"],
        "e07acac20fcd2ee0faca90aa91078ff142724d63"
    );
    assert_eq!(
        manifest["producer"]["llama_core_commit"],
        "b1cd3a914175adedcc976388c7c638d9a2f9a189"
    );
    assert_eq!(
        manifest["producer"]["decision_transcript_llm_patch_file"],
        "deepseek_v4_b10222_decision_transcript_llm.patch"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(LLM_PATCH)),
        manifest["producer"]["decision_transcript_llm_patch_sha256"]
            .as_str()
            .unwrap()
    );
    assert_eq!(
        manifest["producer"]["decision_transcript_llama_core_patch_file"],
        "deepseek_v4_b10222_decision_transcript_llama_core.patch"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(LLAMA_CORE_PATCH)),
        manifest["producer"]["decision_transcript_llama_core_patch_sha256"]
            .as_str()
            .unwrap()
    );
    assert_eq!(
        manifest["request"]["pattern_token_ids"],
        serde_json::json!([35, 201, 200, 34])
    );
    assert_eq!(manifest["request"]["context_size"], 4_096);
    assert_eq!(manifest["request"]["cache_type"], "F16");

    let vectors = [
        (
            "deepseek_v4_pattern35_201_200_34_pre_twenty_fourth_hca_position3070_b10222.f32",
            POSITION_3070_SINGLETON,
            34,
        ),
        (
            "deepseek_v4_pattern35_201_200_34_pre_twenty_fourth_hca_position3070_b10222_batched.f32",
            POSITION_3070_BATCHED,
            34,
        ),
        (
            "deepseek_v4_pattern35_201_200_34_x768_position3071_b10222.f32",
            POSITION_3071_SINGLETON,
            35,
        ),
        (
            "deepseek_v4_pattern35_201_200_34_x768_position3071_b10222_batched.f32",
            POSITION_3071_BATCHED,
            35,
        ),
        (
            "deepseek_v4_pattern35_201_200_34_x768_then35_position3072_b10222.f32",
            POSITION_3072_SINGLETON,
            201,
        ),
        (
            "deepseek_v4_pattern35_201_200_34_x768_then35_position3072_b10222_batched.f32",
            POSITION_3072_BATCHED,
            201,
        ),
    ];
    let manifest_vectors = manifest["vectors"].as_array().unwrap();
    assert_eq!(manifest_vectors.len(), vectors.len());
    for (file, bytes, expected_argmax) in vectors {
        let entry = manifest_vectors
            .iter()
            .find(|entry| entry["file"] == file)
            .unwrap();
        assert_eq!(bytes.len(), 129_280 * std::mem::size_of::<f32>());
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            entry["sha256"].as_str().unwrap()
        );
        let values = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|value| value.is_finite()));
        let argmax = values
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0;
        assert_eq!(argmax, expected_argmax);
        assert_eq!(entry["argmax_token_id"], expected_argmax);
        assert_eq!(entry["fresh_process_repeats"], 2);
        assert_eq!(entry["repeat_vectors_byte_identical"], true);
        let mode = entry["prompt_decode_mode"].as_str().unwrap();
        for command_field in ["capture_command", "repeat_command"] {
            let command = entry[command_field].as_str().unwrap();
            assert!(command.contains("PATCHED_LLM --model MODEL --context 4096"));
            assert!(command.contains("--batch-size 512 --ubatch-size 128"));
            assert!(command.contains("--repeat 1"));
            assert_eq!(command.contains("--prompt-geometry"), mode == "singleton");
        }
        assert!(
            entry["capture_command"]
                .as_str()
                .unwrap()
                .contains("--snapshot OUT ")
        );
        assert!(
            entry["repeat_command"]
                .as_str()
                .unwrap()
                .contains("--snapshot OUT_REPEAT ")
        );
    }

    let transcripts = [
        (
            "deepseek_v4_position3070_native_decisions.json",
            NATIVE_SOURCE,
            "native_packed_prefix_singleton_target",
            "repeat_payloads_byte_identical",
            None,
        ),
        (
            "deepseek_v4_position3070_b10222_batched_decisions.json",
            BATCHED_SOURCE,
            "b10222_batched_prefix_singleton_target",
            "repeat_tensor_payloads_byte_identical",
            Some("560effbe77a6a90a5c986b1d816455540a03a75df500c8312b6613253c1bc11c"),
        ),
        (
            "deepseek_v4_position3070_b10222_singleton_decisions.json",
            SINGLETON_SOURCE,
            "b10222_singleton_prefix_singleton_target",
            "repeat_tensor_payloads_byte_identical",
            Some("16cc992f92183efaeb25be32ea63c116f5e43bdea37886941fb8f58d113fd3b0"),
        ),
    ];
    let manifest_transcripts = manifest["decision_transcripts"].as_array().unwrap();
    assert_eq!(manifest_transcripts.len(), transcripts.len());
    for (file, source, schedule, repeat_field, canonical_tensor_sha256) in transcripts {
        let entry = manifest_transcripts
            .iter()
            .find(|entry| entry["file"] == file)
            .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(source.as_bytes())),
            entry["sha256"].as_str().unwrap()
        );
        assert_eq!(entry["schedule"], schedule);
        assert_eq!(entry["fresh_process_repeats"], 2);
        assert_eq!(entry[repeat_field], true);
        if let Some(expected) = canonical_tensor_sha256 {
            assert_eq!(entry["canonical_tensor_sha256"], expected);
        }
        let capture_command = entry["capture_command"].as_str().unwrap();
        let repeat_command = entry["repeat_command"].as_str().unwrap();
        assert!(!capture_command.is_empty());
        assert!(!repeat_command.is_empty());
        if schedule.starts_with("b10222_") {
            assert!(capture_command.contains("--batch-size 512 --ubatch-size 128"));
            assert!(repeat_command.contains("--batch-size 512 --ubatch-size 128"));
            assert_eq!(
                capture_command.contains("--prompt-geometry"),
                schedule.contains("singleton_prefix")
            );
            assert_eq!(
                repeat_command.contains("--prompt-geometry"),
                schedule.contains("singleton_prefix")
            );
            assert!(capture_command.contains("--decision-transcript TRANSCRIPT "));
            assert!(repeat_command.contains("--decision-transcript TRANSCRIPT_REPEAT "));
        }
    }
    assert_eq!(
        manifest["mechanism_gate"]["native_layer4_selected_only"],
        serde_json::json!([378, 381])
    );
    assert_eq!(
        manifest["mechanism_gate"]["b10222_layer4_selected_only"],
        serde_json::json!([383, 384])
    );
    assert_eq!(
        manifest["mechanism_gate"]["b10222_first_schedule_csa_set_difference_layer"],
        6
    );
    let native_hashes = &manifest["native_repeat_hashes"];
    for (field, expected) in [
        (
            "position3070_singleton_target",
            "2479973378fe4befe8304dc89b4a836274ad7ccb3fbda4b3e4f0c8c63ad0defc",
        ),
        (
            "position3071_packed",
            "ee1a5814897a2d89bd884e6089cf40803939ac6140fca95906ce2349f9485126",
        ),
        (
            "position3071_split",
            "f3be04a3823f7008e6735c81c35ae1ad8c6d68dd72698f77082f1c2843c41b25",
        ),
        (
            "position3072_packed",
            "e28ab0a9dd3d8bcd3eab8007334f1dfe5d63a73cabb0701db8159ba3a8e156da",
        ),
        (
            "position3072_split",
            "d85c2d5698ebc537ae39d2ced5d34892c4340250fe4811d031a55013118d4db1",
        ),
    ] {
        assert_eq!(native_hashes[field], expected);
    }
    assert_eq!(native_hashes["fresh_process_repeats"], 2);
    assert_eq!(native_hashes["repeat_vectors_byte_identical"], true);
    assert_eq!(native_hashes["packed_64_128_vectors_byte_identical"], true);
}
