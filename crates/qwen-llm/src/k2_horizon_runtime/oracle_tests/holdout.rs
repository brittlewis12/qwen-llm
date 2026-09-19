use super::*;

mod online;
mod ranking;
mod runner;
mod v2;

const POLICY_BYTES: &[u8] =
    include_bytes!("../../../../../scripts/reference/k2/holdout-256-v1.json");
const POLICY_SHA256: &str = "65a517ad27cab60a7a38989ce8cf3499902940fb94f82c26d83dcca86f5f6889";

fn policy() -> serde_json::Value {
    bound_policy(POLICY_BYTES, POLICY_SHA256)
}

fn bound_policy(bytes: &[u8], expected_hash: &str) -> serde_json::Value {
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        expected_hash,
        "frozen holdout changed; do not retune an executed holdout"
    );
    serde_json::from_slice(bytes).unwrap()
}

fn inputs(source: &GgufFile) -> Vec<(String, u32, Vec<u32>, bool)> {
    let policy = policy();
    inputs_for(source, &policy)
}

fn inputs_for(source: &GgufFile, policy: &serde_json::Value) -> Vec<(String, u32, Vec<u32>, bool)> {
    assert_eq!(source.shards.len(), 1, "frozen artifact is a single GGUF");
    let stamps = source.revalidate_retained_shard_stamps().unwrap();
    assert_eq!(
        file_sha256(&source.shards[0].path),
        policy["model_sha256"],
        "wrong frozen checkpoint"
    );
    assert_eq!(source.revalidate_retained_shard_stamps().unwrap(), stamps);
    assert_eq!(
        format!(
            "{:016x}",
            crate::runtime::tokenizer_metadata_identity(source)
        ),
        policy["tokenizer_metadata_id"]
    );
    let tokenizer = NativeTokenizer::from_gguf(source).unwrap();
    let mut cases = Vec::new();
    for (index, corpus) in policy["corpora"].as_array().unwrap().iter().enumerate() {
        let name = corpus["name"].as_str().unwrap();
        let text = corpus["text"].as_str().unwrap();
        let mut tokens = tokenizer
            .encode(text, true)
            .unwrap()
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect::<Vec<_>>();
        assert!(
            tokens.len() >= 256,
            "{name} has only {} tokens; no repeated filler allowed",
            tokens.len()
        );
        assert_eq!(
            tokens.len() as u64,
            corpus["native_token_count"].as_u64().unwrap(),
            "{name} tokenizer/count drift"
        );
        eprintln!(
            "frozen holdout {name}: {} native tokens; fixed 256-token fixture prefix",
            tokens.len()
        );
        tokens.truncate(256);
        let token_bytes = tokens
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            format!("{:x}", Sha256::digest(&token_bytes)),
            corpus["prefix_sha256_i32le"],
            "{name} token-ID drift"
        );
        for base in policy["bases"].as_array().unwrap() {
            let base = u32::try_from(base.as_u64().unwrap()).unwrap();
            let selected = u64::from(base)
                == policy["continuation"]["bases_by_corpus"][index]
                    .as_u64()
                    .unwrap();
            cases.push((name.to_owned(), base, tokens.clone(), selected));
        }
    }
    cases
}

#[test]
fn frozen_holdout_policy_is_bounded_and_unchanged() {
    let p = policy();
    assert_eq!(p["schema"], "k2.guarded-context-holdout.v1");
    assert_eq!(p["capacity"], 256);
    assert_eq!(p["bases"], json!([0, 37, 8191]));
    assert_eq!(p["corpora"].as_array().unwrap().len(), 4);
    assert_eq!(
        p["boundaries"],
        json!([
            1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256
        ])
    );
    assert_eq!(
        p["logits"],
        json!({"top1_equal":true,"max_abs_exclusive":0.5,"rmse_exclusive":0.05,"cosine_exclusive_min":0.9995})
    );
    assert_eq!(
        p["distribution"],
        json!({"kl_reference_to_actual_exclusive":0.0001,"total_variation_exclusive":0.002,"centered_rmse_exclusive":0.03})
    );
    assert_eq!(
        p["captures"],
        json!({"layers":[0,11,23,35],"cosine_exclusive_min":0.99999,"relative_l2_exclusive":0.005})
    );
    assert_eq!(
        p["continuation"],
        json!({"prefix_length":241,"argmax_transitions":15,"tie_break":"highest token ID",
        "eos_policy":"fixed mathematical argmax trajectory including EOS if produced; serving EOS-stop semantics are tested separately",
        "bases_by_corpus":[0,37,8191,0]})
    );
    assert_eq!(
        p["model_sha256"],
        "5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf"
    );
    assert_eq!(p["tokenizer_metadata_id"], "51ebd8140ea2abd9");
    assert_eq!(
        p["corpora"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["native_token_count"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![349, 440, 349, 371]
    );
    assert_eq!(
        p["continuation"]["prefix_length"].as_u64().unwrap()
            + p["continuation"]["argmax_transitions"].as_u64().unwrap(),
        256
    );
    assert_eq!(p["captures"]["layers"], json!([0, 11, 23, 35]));
    let boundaries = p["boundaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(boundaries.first(), Some(&1));
    assert_eq!(boundaries.last(), Some(&256));
    assert!(boundaries.windows(2).all(|pair| pair[0] < pair[1]));
    let mut names = std::collections::HashSet::new();
    for corpus in p["corpora"].as_array().unwrap() {
        assert!(names.insert(corpus["name"].as_str().unwrap()));
        assert!(corpus["text"].as_str().unwrap().len() > 256);
    }
}

#[test]
#[ignore = "CPU artifact-hash/tokenizer only; requires K2_GGUF, no model forward or Metal"]
fn cpu_frozen_holdout_prefixes_fit_native_tokenizer() {
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let cases = inputs(&source);
    assert_eq!(cases.len(), 12);
    assert_eq!(cases.iter().filter(|case| case.3).count(), 4);
    for (_, _, tokens, _) in cases {
        assert_eq!(tokens.len(), 256);
        assert_eq!(tokens[0], 0);
    }
}
