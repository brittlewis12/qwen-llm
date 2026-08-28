use super::*;
use crate::qwen4exp_metal::Qwen4ExpHcPackedProjectionRecord;
use crate::tokenizer::{
    QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256, qwen4exp_tokenizer_identity_sha256,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const FIXTURE_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../../docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/fixtures.json");
const SCORER_SOURCE_BYTES: &[u8] = include_bytes!("qwen4exp_selected_quality.rs");
const FIXTURE_MANIFEST_SHA256: &str =
    "689e94bf135eac09f50cbf88de046004301f7937d4ded2d5cd4434ef7e45ced1";
const PACKET_ID: &str = "2026-08-28-qwen4exp-selected-quality-v2";
const FIXTURE_SCHEMA: &str = "qwen4exp-selected-quality-fixtures";
const SELECTED_START: usize = 2_051;
const CONTINUATION_TOKENS: usize = 96;
const VOCAB_SIZE: usize = 248_320;
const LOCAL_OPERATION_COUNT: usize = 78;
const TOTAL_OPERATION_COUNT: usize = 98;
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const GGUF_RS_COMMIT: &str = "3a92c518bce43959686bef1093b31b4067502d2b";
const GGUF_RS_TREE: &str = "c4fd676301c9b9ac58a8ef1441717384afa4fac9";
const LLAMA_CPP_RS_COMMIT: &str = "0f1868b3b52dea227c16b5707578f935042cf668";
const LLAMA_CPP_SYS_TREE: &str = "bf14ff07169dc0b52112146c1e35918a400234b5";

#[derive(Clone, Debug, Deserialize)]
struct FixtureManifest {
    schema: String,
    schema_version: u64,
    packet_id: String,
    status: String,
    preregistration: SourceIdentity,
    generator: GeneratorIdentity,
    acquisition_model_lock: ModelLock,
    tokenizer: TokenizerLock,
    corpus: Value,
    execution: ExecutionContract,
    natural_fixtures: Vec<NaturalFixture>,
    scope_control: ScopeFixture,
    retrieval_fixtures: Vec<RetrievalFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceIdentity {
    path: String,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct GeneratorIdentity {
    path: String,
    bytes: usize,
    sha256: String,
    source_plan_path: String,
    source_plan_bytes: usize,
    source_plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelLock {
    repository: String,
    revision: String,
    quant: String,
    fixture_generation_dependency: String,
    verification_phase: String,
    shard_manifest_schema: String,
    shard_manifest_domain_utf8: String,
    shard_manifest_sha256: String,
    shards: Vec<ModelShardLock>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelShardLock {
    index: usize,
    filename: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TokenizerLock {
    identity_schema: String,
    identity_sha256: String,
    vocab_size: usize,
    producer_stop_token_ids: Vec<u32>,
    model: String,
    pretokenizer: String,
    chat_template_sha256: String,
    add_special_tokens: bool,
    fixture_generation_qualification: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ExecutionContract {
    local_arms: Vec<String>,
    orders: Vec<String>,
    each_permutation_count: usize,
    operation_plan: Vec<Operation>,
    support_runs: Value,
    llama_cpp: LlamaCppLock,
    bootstrap: Value,
}

#[derive(Clone, Debug, Deserialize)]
struct LlamaCppLock {
    arm: String,
    repository: String,
    commit: String,
    support_pull_request: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Operation {
    ordinal: usize,
    phase: String,
    fixture_id: String,
    mode: String,
    arm: String,
    ordering_key_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct DocumentIdentity {
    source_ordinal: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct TokenRecord {
    path: String,
    dtype: String,
    byte_order: String,
    token_count: usize,
    bytes: usize,
    sha256: String,
    sha256_raw_i32le: String,
    digest_domain_utf8: String,
    sha256_domain_i32le: String,
}

#[derive(Clone, Debug, Deserialize)]
struct NaturalFixture {
    fixture_id: String,
    document: DocumentIdentity,
    prompt_token_count: usize,
    selected_suffix_tokens: usize,
    continuation_token_count: usize,
    arm_order: String,
    open_greedy_sentinel: bool,
    reverse_replay: bool,
    tokens: TokenRecord,
    prompt_sha256_domain_i32le: String,
    continuation_sha256_domain_i32le: String,
    terminal_feed_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct ScopeFixture {
    fixture_id: String,
    document: DocumentIdentity,
    prompt_token_count: usize,
    selected_rows_expected: usize,
    arm_order: String,
    tokens: TokenRecord,
}

#[derive(Clone, Debug, Deserialize)]
struct RetrievalFixture {
    fixture_id: String,
    kind: String,
    document: DocumentIdentity,
    evidence: Vec<RetrievalEvidence>,
    symbol_exclusion_audit: Vec<SymbolAudit>,
    answer_text: String,
    answer_text_sha256_utf8: String,
    answer_token_ids: Vec<u32>,
    answer_token_count: usize,
    answer_token_ids_sha256_raw_i32le: String,
    decoy: RetrievalDecoy,
    producer_stop_token_ids: Vec<u32>,
    max_generated_tokens: usize,
    exact_match: ExactMatchContract,
    tokens: TokenRecord,
    arm_order: String,
}

#[derive(Clone, Debug, Deserialize)]
struct RetrievalEvidence {
    branch: String,
    role: String,
    target_token_index: usize,
    realized_token_span: [usize; 2],
    payload_sha256_utf8: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SymbolAudit {
    role: String,
    utf8_sha256: String,
    source_document_occurrences: usize,
    expected_prompt_occurrences: usize,
    actual_prompt_occurrences: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RetrievalDecoy {
    key: String,
    relay: Option<String>,
    answer_text: String,
    answer_text_sha256_utf8: String,
    answer_token_ids: Vec<u32>,
    answer_token_count: usize,
    answer_token_ids_sha256_raw_i32le: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ExactMatchContract {
    normalization: String,
    require_answer_from_generated_token_zero: bool,
    require_complete_answer: bool,
    require_immediate_following_producer_stop: bool,
}

struct ValidatedFixtures {
    root: PathBuf,
    manifest: FixtureManifest,
    tokens: BTreeMap<String, Vec<u32>>,
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn token_domain(fixture_id: &str, role: &str, count: usize) -> String {
    format!(
        "qwen4exp-selected-quality-token-ids-i32le-v1\0fixture_id={fixture_id}\nrole={role}\ncount={count}\n"
    )
}

fn read_i32le(bytes: &[u8]) -> Vec<i32> {
    assert!(bytes.len().is_multiple_of(std::mem::size_of::<i32>()));
    bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn validate_relative_path(path: &str) {
    let path = Path::new(path);
    assert!(!path.is_absolute(), "fixture path must be relative");
    assert!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "fixture path contains a non-normal component: {}",
        path.display()
    );
}

fn read_token_record(
    root: &Path,
    fixture_id: &str,
    role: &str,
    record: &TokenRecord,
    expected_vocab: usize,
) -> Vec<u32> {
    validate_relative_path(&record.path);
    assert!(record.path.starts_with("tokens/"));
    assert_eq!(record.dtype, "i32");
    assert_eq!(record.byte_order, "little");
    assert_eq!(record.bytes, record.token_count * 4);
    let bytes = std::fs::read(root.join(&record.path)).unwrap();
    assert_eq!(bytes.len(), record.bytes, "{fixture_id} token bytes");
    let raw_sha256 = sha256_bytes(&bytes);
    assert_eq!(raw_sha256, record.sha256, "{fixture_id} raw SHA-256");
    assert_eq!(
        raw_sha256, record.sha256_raw_i32le,
        "{fixture_id} raw i32 SHA-256"
    );
    let signed = read_i32le(&bytes);
    assert_eq!(signed.len(), record.token_count);
    let expected_domain = token_domain(fixture_id, role, signed.len());
    assert_eq!(record.digest_domain_utf8, expected_domain);
    assert_eq!(
        sha256_i32_le(expected_domain.as_bytes(), &signed),
        record.sha256_domain_i32le,
        "{fixture_id} domain token SHA-256"
    );
    signed
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            let token = u32::try_from(token)
                .unwrap_or_else(|_| panic!("{fixture_id} token {index} is negative: {token}"));
            assert!(
                (token as usize) < expected_vocab,
                "{fixture_id} token {index} is outside vocabulary: {token}"
            );
            token
        })
        .collect()
}

fn validate_arm_order(order: &str) {
    assert_eq!(order.len(), 3);
    assert_eq!(
        order.chars().collect::<BTreeSet<_>>(),
        ['A', 'B', 'C'].into()
    );
}

fn append_expected_operations(
    expected: &mut Vec<(String, String, String, String)>,
    phase: &str,
    fixture_id: &str,
    mode: &str,
    order: &str,
) {
    validate_arm_order(order);
    expected.extend(order.chars().map(|arm| {
        (
            phase.to_string(),
            fixture_id.to_string(),
            mode.to_string(),
            arm.to_string(),
        )
    }));
}

fn validate_operation_plan(manifest: &FixtureManifest) {
    let mut expected = Vec::new();
    append_expected_operations(
        &mut expected,
        "scope_control",
        &manifest.scope_control.fixture_id,
        "prefill_endpoint_replay",
        &manifest.scope_control.arm_order,
    );
    for fixture in &manifest.natural_fixtures {
        append_expected_operations(
            &mut expected,
            "natural_semantic",
            &fixture.fixture_id,
            "teacher_forced_nll_96",
            &fixture.arm_order,
        );
    }
    for fixture in manifest
        .natural_fixtures
        .iter()
        .filter(|fixture| fixture.open_greedy_sentinel)
    {
        append_expected_operations(
            &mut expected,
            "open_greedy",
            &fixture.fixture_id,
            "greedy_32_or_stop",
            &fixture.arm_order,
        );
    }
    for fixture in &manifest.retrieval_fixtures {
        append_expected_operations(
            &mut expected,
            "retrieval_semantic",
            &fixture.fixture_id,
            "answer_nll_and_exact_prefix",
            &fixture.arm_order,
        );
    }
    let reverse = manifest
        .natural_fixtures
        .iter()
        .find(|fixture| fixture.reverse_replay)
        .unwrap();
    let reverse_order = reverse.arm_order.chars().rev().collect::<String>();
    append_expected_operations(
        &mut expected,
        "reverse_replay",
        &reverse.fixture_id,
        "teacher_forced_nll_96_replay",
        &reverse_order,
    );
    assert_eq!(expected.len(), LOCAL_OPERATION_COUNT);
    assert_eq!(
        manifest.execution.operation_plan.len(),
        TOTAL_OPERATION_COUNT
    );
    for (ordinal, operation) in manifest.execution.operation_plan.iter().enumerate() {
        assert_eq!(operation.ordinal, ordinal);
        if ordinal < LOCAL_OPERATION_COUNT {
            let expected = &expected[ordinal];
            assert_eq!(
                (
                    &operation.phase,
                    &operation.fixture_id,
                    &operation.mode,
                    &operation.arm,
                ),
                (&expected.0, &expected.1, &expected.2, &expected.3)
            );
            assert!(operation.ordering_key_sha256.is_none());
        } else {
            assert_eq!(operation.phase, "llama_cpp_triangulation");
            assert_eq!(operation.arm, "D");
            assert!(is_lower_hex(
                operation.ordering_key_sha256.as_deref().unwrap(),
                64
            ));
        }
    }
    let local_fixture_ids = manifest
        .natural_fixtures
        .iter()
        .map(|fixture| fixture.fixture_id.as_str())
        .chain(
            manifest
                .retrieval_fixtures
                .iter()
                .map(|fixture| fixture.fixture_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    let llama_fixture_ids = manifest.execution.operation_plan[LOCAL_OPERATION_COUNT..]
        .iter()
        .map(|operation| operation.fixture_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(llama_fixture_ids, local_fixture_ids);
}

fn validate_source_identity(root: &Path, identity: &SourceIdentity) {
    validate_relative_path(&identity.path);
    let bytes = std::fs::read(root.join(&identity.path)).unwrap();
    assert_eq!(bytes.len(), identity.bytes, "{} bytes", identity.path);
    assert_eq!(
        sha256_bytes(&bytes),
        identity.sha256,
        "{} SHA-256",
        identity.path
    );
}

fn validate_model_lock(model: &ModelLock) {
    assert_eq!(model.repository, "unsloth/Qwen3.8-Flash-Next-GGUF");
    assert_eq!(model.revision, "8bdc666649440e9bdc97e16f3f75782c98478ff5");
    assert_eq!(model.quant, "UD-Q3_K_XL");
    assert!(model.fixture_generation_dependency.contains("tokenizer"));
    assert!(model.verification_phase.contains("hash every local shard"));
    assert_eq!(
        model.shard_manifest_schema,
        "qwen4exp-release-model-shard-manifest-v1"
    );
    assert_eq!(model.shards.len(), 3);
    let mut domain = String::from("qwen4exp-release-model-shard-manifest-v1\0");
    for (index, shard) in model.shards.iter().enumerate() {
        assert_eq!(shard.index, index);
        assert!(shard.filename.ends_with(".gguf"));
        assert!(shard.bytes > 0);
        assert!(is_lower_hex(&shard.sha256, 64));
        domain.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            shard.index, shard.filename, shard.bytes, shard.sha256
        ));
    }
    assert_eq!(model.shard_manifest_domain_utf8, domain);
    assert_eq!(sha256_bytes(domain.as_bytes()), model.shard_manifest_sha256);
}

fn validate_execution_contract(manifest: &FixtureManifest) {
    assert_eq!(
        manifest.execution.local_arms,
        ["A_default_safe", "B_generic_selected", "C_f32_hc_down"]
    );
    assert_eq!(manifest.execution.orders.len(), 12);
    let mut permutation_counts = BTreeMap::<String, usize>::new();
    for order in &manifest.execution.orders {
        validate_arm_order(order);
        *permutation_counts.entry(order.clone()).or_default() += 1;
    }
    assert_eq!(permutation_counts.len(), 6);
    assert!(
        permutation_counts
            .values()
            .all(|&count| count == manifest.execution.each_permutation_count)
    );
    assert_eq!(manifest.execution.each_permutation_count, 2);
    assert_eq!(manifest.execution.support_runs["included"], false);
    assert_eq!(
        manifest.execution.llama_cpp.arm,
        "D_same_manifest_tokens_separate_order"
    );
    assert_eq!(
        manifest.execution.llama_cpp.repository,
        "ggml-org/llama.cpp"
    );
    assert_eq!(
        manifest.execution.llama_cpp.commit,
        "6c84c7d5d8833c6e0df69628f75a0f599797934e"
    );
    assert_eq!(manifest.execution.llama_cpp.support_pull_request, 27_742);
    let bootstrap = &manifest.execution.bootstrap;
    assert_eq!(bootstrap["unit"], "document");
    assert_eq!(bootstrap["strata"], json!([2179, 2563, 3075, 4099]));
    assert_eq!(bootstrap["documents_per_stratum"], 3);
    assert_eq!(bootstrap["draws"], 100_000);
    assert_eq!(bootstrap["rng"], "splitmix64-rejection-u64-v1");
    assert_eq!(bootstrap["seed_u64"], 0x38f1_a9c5_d204_7e61_u64);
    assert_eq!(bootstrap["quantiles"]["candidate_vs_incumbent"], 0.975);
    assert_eq!(bootstrap["quantiles"]["challenger_vs_generic"], 0.95);
    assert!(
        bootstrap["draw_order"]
            .as_str()
            .unwrap()
            .contains("ascending stratum")
    );
    assert!(
        bootstrap["bounded_mapping"]
            .as_str()
            .unwrap()
            .contains("modulo 3")
    );
    assert!(
        bootstrap["quantile_convention"]
            .as_str()
            .unwrap()
            .contains("no interpolation")
    );
    validate_operation_plan(manifest);
}

fn validate_fixture_set() -> ValidatedFixtures {
    assert_eq!(
        sha256_bytes(FIXTURE_MANIFEST_BYTES),
        FIXTURE_MANIFEST_SHA256,
        "embedded fixture manifest identity"
    );
    let root = repository_root();
    let fixture_root = root.join("docs/bench/2026-08-28-qwen4exp-selected-quality-prereg");
    assert_eq!(
        std::fs::read(fixture_root.join("fixtures.json")).unwrap(),
        FIXTURE_MANIFEST_BYTES
    );
    let manifest: FixtureManifest = serde_json::from_slice(FIXTURE_MANIFEST_BYTES).unwrap();
    assert_eq!(manifest.schema, FIXTURE_SCHEMA);
    assert_eq!(manifest.schema_version, 2);
    assert_eq!(manifest.packet_id, PACKET_ID);
    assert_eq!(manifest.status, "fixtures_frozen_not_acquired");
    validate_source_identity(&root, &manifest.preregistration);
    validate_source_identity(
        &root,
        &SourceIdentity {
            path: manifest.generator.path.clone(),
            bytes: manifest.generator.bytes,
            sha256: manifest.generator.sha256.clone(),
        },
    );
    assert!(manifest.generator.source_plan_path.starts_with("target/"));
    assert!(manifest.generator.source_plan_bytes > 0);
    assert!(is_lower_hex(&manifest.generator.source_plan_sha256, 64));
    validate_model_lock(&manifest.acquisition_model_lock);
    assert_eq!(
        manifest.tokenizer.identity_schema,
        "qwen4exp-tokenizer-identity-v2"
    );
    assert_eq!(
        manifest.tokenizer.identity_sha256,
        "86a6193d6a6c9b43a71a207765f85fb3904addb3bba6dcd50b50bef076e5e89f"
    );
    assert_eq!(manifest.tokenizer.vocab_size, VOCAB_SIZE);
    assert_eq!(manifest.tokenizer.producer_stop_token_ids, [248_046]);
    assert_eq!(manifest.tokenizer.model, "gpt2");
    assert_eq!(manifest.tokenizer.pretokenizer, "qwen35");
    assert!(is_lower_hex(&manifest.tokenizer.chat_template_sha256, 64));
    assert!(!manifest.tokenizer.add_special_tokens);
    assert!(
        manifest
            .tokenizer
            .fixture_generation_qualification
            .contains("model weight bytes are not fixture inputs")
    );
    assert_eq!(manifest.corpus["source"]["config"], "wikitext-2-raw-v1");
    assert_eq!(
        manifest.corpus["selection"]["repository_exclusion_audit"]["passed"],
        true
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_exclusion_audit"]["repository_commit"],
        "f9bbbdc0228e1029225fbdb6d4e83a14b846415e"
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_exclusion_audit"]["token_fixture_count"],
        1
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_exclusion_audit"]["window_audit"]["window_count"],
        44_142
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_token_fixture_exclusion_audit"]["passed"],
        true
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_token_fixture_exclusion_audit"]["current_fixture_count"],
        21
    );
    assert_eq!(
        manifest.corpus["selection"]["repository_token_fixture_exclusion_audit"]["maximum_common_five_grams"],
        0
    );
    assert_eq!(
        manifest.corpus["selection"]["near_duplicate_audit"]["passed"],
        true
    );
    validate_execution_contract(&manifest);

    assert_eq!(manifest.natural_fixtures.len(), 12);
    assert_eq!(manifest.retrieval_fixtures.len(), 8);
    assert_eq!(manifest.scope_control.fixture_id, "scope-n2051");
    assert_eq!(manifest.scope_control.prompt_token_count, SELECTED_START);
    assert_eq!(manifest.scope_control.selected_rows_expected, 0);
    validate_arm_order(&manifest.scope_control.arm_order);

    let mut tokens = BTreeMap::new();
    let mut token_paths = BTreeSet::new();
    let mut document_ordinals = BTreeSet::new();
    let mut shape_counts = BTreeMap::<usize, usize>::new();
    let mut open_shape_counts = BTreeMap::<usize, usize>::new();
    let mut reverse_count = 0;
    for fixture in &manifest.natural_fixtures {
        assert!(document_ordinals.insert(fixture.document.source_ordinal));
        assert!(token_paths.insert(fixture.tokens.path.clone()));
        assert!(
            tokens
                .insert(
                    fixture.fixture_id.clone(),
                    read_token_record(
                        &fixture_root,
                        &fixture.fixture_id,
                        "prompt_then_continuation",
                        &fixture.tokens,
                        manifest.tokenizer.vocab_size,
                    ),
                )
                .is_none()
        );
        let fixture_tokens = &tokens[&fixture.fixture_id];
        assert_eq!(
            fixture_tokens.len(),
            fixture.prompt_token_count + CONTINUATION_TOKENS
        );
        assert_eq!(fixture.continuation_token_count, CONTINUATION_TOKENS);
        assert_eq!(
            fixture.selected_suffix_tokens,
            fixture.prompt_token_count - SELECTED_START
        );
        assert!(fixture.selected_suffix_tokens.is_multiple_of(128));
        assert_eq!(
            sha256_u32_le(
                token_domain(&fixture.fixture_id, "prompt", fixture.prompt_token_count).as_bytes(),
                &fixture_tokens[..fixture.prompt_token_count],
            ),
            fixture.prompt_sha256_domain_i32le
        );
        assert_eq!(
            sha256_u32_le(
                token_domain(&fixture.fixture_id, "continuation", CONTINUATION_TOKENS).as_bytes(),
                &fixture_tokens[fixture.prompt_token_count..],
            ),
            fixture.continuation_sha256_domain_i32le
        );
        assert_eq!(
            *fixture_tokens.last().unwrap(),
            fixture.terminal_feed_token_id
        );
        validate_arm_order(&fixture.arm_order);
        *shape_counts.entry(fixture.prompt_token_count).or_default() += 1;
        if fixture.open_greedy_sentinel {
            *open_shape_counts
                .entry(fixture.prompt_token_count)
                .or_default() += 1;
        }
        reverse_count += usize::from(fixture.reverse_replay);
    }
    assert_eq!(
        shape_counts,
        [(2_179, 3), (2_563, 3), (3_075, 3), (4_099, 3)].into()
    );
    assert_eq!(
        open_shape_counts,
        [(2_179, 1), (2_563, 1), (3_075, 1), (4_099, 1)].into()
    );
    assert_eq!(reverse_count, 1);

    assert!(document_ordinals.insert(manifest.scope_control.document.source_ordinal));
    assert!(token_paths.insert(manifest.scope_control.tokens.path.clone()));
    let scope_tokens = read_token_record(
        &fixture_root,
        &manifest.scope_control.fixture_id,
        "prompt",
        &manifest.scope_control.tokens,
        manifest.tokenizer.vocab_size,
    );
    assert_eq!(scope_tokens.len(), SELECTED_START);
    assert!(
        tokens
            .insert(manifest.scope_control.fixture_id.clone(), scope_tokens)
            .is_none()
    );

    let mut answer_tokens = BTreeSet::new();
    for fixture in &manifest.retrieval_fixtures {
        assert!(document_ordinals.insert(fixture.document.source_ordinal));
        assert!(token_paths.insert(fixture.tokens.path.clone()));
        let fixture_tokens = read_token_record(
            &fixture_root,
            &fixture.fixture_id,
            "prompt",
            &fixture.tokens,
            manifest.tokenizer.vocab_size,
        );
        assert_eq!(fixture_tokens.len(), 4_099);
        assert!(
            tokens
                .insert(fixture.fixture_id.clone(), fixture_tokens)
                .is_none()
        );
        assert!(matches!(fixture.kind.as_str(), "single_hop" | "two_hop"));
        assert_eq!(fixture.answer_token_count, fixture.answer_token_ids.len());
        assert!((1..=2).contains(&fixture.answer_token_count));
        assert_eq!(
            sha256_bytes(fixture.answer_text.as_bytes()),
            fixture.answer_text_sha256_utf8
        );
        assert_eq!(
            sha256_u32_le(b"", &fixture.answer_token_ids),
            fixture.answer_token_ids_sha256_raw_i32le
        );
        assert!(answer_tokens.insert(fixture.answer_token_ids.clone()));
        assert_eq!(
            fixture.decoy.answer_token_count,
            fixture.decoy.answer_token_ids.len()
        );
        assert_eq!(fixture.decoy.answer_token_count, fixture.answer_token_count);
        assert_eq!(
            sha256_bytes(fixture.decoy.answer_text.as_bytes()),
            fixture.decoy.answer_text_sha256_utf8
        );
        assert_eq!(
            sha256_u32_le(b"", &fixture.decoy.answer_token_ids),
            fixture.decoy.answer_token_ids_sha256_raw_i32le
        );
        assert!(answer_tokens.insert(fixture.decoy.answer_token_ids.clone()));
        assert!(!fixture.decoy.key.is_empty());
        assert_eq!(fixture.decoy.relay.is_some(), fixture.kind == "two_hop");
        assert_eq!(
            fixture.producer_stop_token_ids,
            manifest.tokenizer.producer_stop_token_ids
        );
        assert_eq!(fixture.max_generated_tokens, 8);
        assert_eq!(fixture.exact_match.normalization, "none");
        assert!(fixture.exact_match.require_answer_from_generated_token_zero);
        assert!(fixture.exact_match.require_complete_answer);
        assert!(
            fixture
                .exact_match
                .require_immediate_following_producer_stop
        );
        let expected_evidence = if fixture.kind == "single_hop" { 2 } else { 4 };
        assert_eq!(fixture.evidence.len(), expected_evidence);
        let mut previous_end = 0;
        let mut branches = BTreeMap::<String, usize>::new();
        for evidence in &fixture.evidence {
            assert!(matches!(evidence.branch.as_str(), "target" | "decoy"));
            assert!(matches!(
                evidence.role.as_str(),
                "single_hop_value" | "two_hop_key_to_relay" | "two_hop_relay_to_value"
            ));
            assert_eq!(evidence.target_token_index, evidence.realized_token_span[0]);
            assert!(evidence.realized_token_span[0] >= previous_end);
            assert!(evidence.realized_token_span[1] > evidence.realized_token_span[0]);
            assert!(evidence.realized_token_span[1] <= 4_099);
            assert!(is_lower_hex(&evidence.payload_sha256_utf8, 64));
            previous_end = evidence.realized_token_span[1];
            *branches.entry(evidence.branch.clone()).or_default() += 1;
        }
        assert_eq!(branches["target"], expected_evidence / 2);
        assert_eq!(branches["decoy"], expected_evidence / 2);
        assert_eq!(fixture.symbol_exclusion_audit.len(), 6);
        let symbol_roles = fixture
            .symbol_exclusion_audit
            .iter()
            .map(|audit| audit.role.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            symbol_roles,
            [
                "target_key",
                "target_relay",
                "target_answer",
                "decoy_key",
                "decoy_relay",
                "decoy_answer",
            ]
            .into()
        );
        for audit in &fixture.symbol_exclusion_audit {
            assert!(is_lower_hex(&audit.utf8_sha256, 64));
            assert_eq!(audit.source_document_occurrences, 0);
            assert_eq!(
                audit.actual_prompt_occurrences,
                audit.expected_prompt_occurrences
            );
        }
        validate_arm_order(&fixture.arm_order);
    }
    assert_eq!(document_ordinals.len(), 21);
    assert_eq!(token_paths.len(), 21);
    let actual_token_paths = std::fs::read_dir(fixture_root.join("tokens"))
        .unwrap()
        .map(|entry| format!("tokens/{}", entry.unwrap().file_name().to_string_lossy()))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_token_paths, token_paths);

    ValidatedFixtures {
        root: fixture_root,
        manifest,
        tokens,
    }
}

#[test]
fn selected_quality_fixture_manifest_contract_is_frozen() {
    let fixtures = validate_fixture_set();
    assert_eq!(fixtures.tokens.len(), 21);
    assert!(
        fixtures
            .root
            .ends_with("2026-08-28-qwen4exp-selected-quality-prereg")
    );
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum LocalArm {
    DefaultSafe,
    GenericSelected,
    F32HcDown,
}

impl LocalArm {
    fn from_code(code: &str) -> Self {
        match code {
            "A" => Self::DefaultSafe,
            "B" => Self::GenericSelected,
            "C" => Self::F32HcDown,
            _ => panic!("unknown local arm {code}"),
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::DefaultSafe => "A",
            Self::GenericSelected => "B",
            Self::F32HcDown => "C",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DefaultSafe => "A_default_safe",
            Self::GenericSelected => "B_generic_selected",
            Self::F32HcDown => "C_f32_hc_down",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SnapshotIdentity {
    logits_sha256_f32le: String,
    persistent_state_sha256: String,
    qsa_lengths_sha256: String,
    ple_history_sha256_u32le: String,
    committed_length: usize,
}

impl SnapshotIdentity {
    fn json(&self) -> Value {
        json!({
            "logits_sha256_f32le": self.logits_sha256_f32le,
            "persistent_state_sha256": self.persistent_state_sha256,
            "qsa_lengths_sha256": self.qsa_lengths_sha256,
            "ple_history_sha256_u32le": self.ple_history_sha256_u32le,
            "committed_length": self.committed_length,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReplayIdentity {
    endpoint: SnapshotIdentity,
    terminal: SnapshotIdentity,
    logit_trace_sha256: String,
    logit_trace_rows: usize,
    treatment_sha256: String,
    treatment_records: usize,
}

impl ReplayIdentity {
    fn binding_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"qwen4exp-selected-quality-replay-identity-v1\0");
        for (name, value) in [
            ("endpoint_logits", &self.endpoint.logits_sha256_f32le),
            ("endpoint_state", &self.endpoint.persistent_state_sha256),
            ("endpoint_qsa", &self.endpoint.qsa_lengths_sha256),
            ("endpoint_ple", &self.endpoint.ple_history_sha256_u32le),
            ("terminal_logits", &self.terminal.logits_sha256_f32le),
            ("terminal_state", &self.terminal.persistent_state_sha256),
            ("terminal_qsa", &self.terminal.qsa_lengths_sha256),
            ("terminal_ple", &self.terminal.ple_history_sha256_u32le),
            ("logit_trace", &self.logit_trace_sha256),
            ("treatment", &self.treatment_sha256),
        ] {
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update(value.as_bytes());
        }
        for value in [
            self.endpoint.committed_length,
            self.terminal.committed_length,
            self.logit_trace_rows,
            self.treatment_records,
        ] {
            digest.update((value as u64).to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn json(&self) -> Value {
        json!({
            "endpoint": self.endpoint.json(),
            "terminal": self.terminal.json(),
            "logit_trace_sha256": self.logit_trace_sha256,
            "logit_trace_rows": self.logit_trace_rows,
            "treatment_sha256": self.treatment_sha256,
            "treatment_records": self.treatment_records,
            "binding_sha256": self.binding_sha256(),
        })
    }
}

struct LogitTrace {
    digest: Sha256,
    rows: usize,
}

impl LogitTrace {
    fn new() -> Self {
        let mut digest = Sha256::new();
        digest.update(b"qwen4exp-selected-quality-logit-trace-f32le-v1\0");
        Self { digest, rows: 0 }
    }

    fn push(&mut self, logits: &[f32]) {
        assert_eq!(logits.len(), VOCAB_SIZE);
        assert!(logits.iter().all(|value| value.is_finite()));
        self.digest.update((logits.len() as u64).to_le_bytes());
        for value in logits {
            self.digest.update(value.to_bits().to_le_bytes());
        }
        self.rows += 1;
    }

    fn finish(self) -> (String, usize) {
        (format!("{:x}", self.digest.finalize()), self.rows)
    }
}

struct ScoredToken {
    json: Value,
    nll: f64,
    top1: bool,
}

fn score_token(logits: &[f32], ordinal: usize, target: u32) -> ScoredToken {
    assert_eq!(logits.len(), VOCAB_SIZE);
    assert!((target as usize) < logits.len());
    assert!(logits.iter().all(|value| value.is_finite()));
    let maximum = logits
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = logits
        .iter()
        .copied()
        .map(|value| (f64::from(value) - maximum).exp())
        .sum::<f64>();
    assert!(sum_exp.is_finite() && sum_exp > 0.0);
    let logsumexp = maximum + sum_exp.ln();
    let target_logit = logits[target as usize];
    let nll = logsumexp - f64::from(target_logit);
    assert!(logsumexp.is_finite() && nll.is_finite());
    let prediction = argmax(logits) as u32;
    ScoredToken {
        json: json!({
            "ordinal": ordinal,
            "target_token_id": target,
            "target_logit_f32": target_logit,
            "logsumexp_f64": logsumexp,
            "nll_f64": nll,
            "argmax_token_id": prediction,
            "top1": prediction == target,
        }),
        nll,
        top1: prediction == target,
    }
}

fn snapshot_identity(
    runner: &Qwen4ExpTextRunner<'_, '_, '_>,
    logits: &[f32],
    phase: &str,
) -> SnapshotIdentity {
    let state = snapshot_persistent_state(runner);
    let qsa_lengths = runner.workspace.qsa_committed_lengths();
    let ple_history = runner.workspace.ple_prior_tokens();
    SnapshotIdentity {
        logits_sha256_f32le: sha256_f32_bits(
            format!("qwen4exp-selected-quality-{phase}-logits-f32le-v1\0").as_bytes(),
            logits,
        ),
        persistent_state_sha256: sha256_state_bytes(
            format!("qwen4exp-selected-quality-{phase}-persistent-state-v1\0").as_bytes(),
            &state,
        ),
        qsa_lengths_sha256: sha256_qsa_lengths(
            format!("qwen4exp-selected-quality-{phase}-qsa-lengths-v1\0").as_bytes(),
            &qsa_lengths,
        ),
        ple_history_sha256_u32le: sha256_u32_le(
            format!("qwen4exp-selected-quality-{phase}-ple-history-u32le-v1\0").as_bytes(),
            ple_history,
        ),
        committed_length: runner.next_position(),
    }
}

fn reset_and_zero(runner: &mut Qwen4ExpTextRunner<'_, '_, '_>) {
    runner.reset().unwrap();
    assert_eq!(runner.next_position(), 0);
    zero_persistent_state(runner);
    assert!(
        runner
            .workspace
            .qsa_committed_lengths()
            .iter()
            .all(|(_, length)| *length == 0)
    );
}

fn with_local_arm<R>(
    arm: LocalArm,
    prompt_tokens: usize,
    f: impl FnOnce() -> R,
) -> (R, Vec<Qwen4ExpHcPackedProjectionRecord>) {
    match arm {
        LocalArm::DefaultSafe => {
            let _selected = Qwen4ExpPackedSelectedQsaOverride::set(false);
            (f(), Vec::new())
        }
        LocalArm::GenericSelected => {
            let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
            (f(), Vec::new())
        }
        LocalArm::F32HcDown if prompt_tokens == SELECTED_START => {
            let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
            (f(), Vec::new())
        }
        LocalArm::F32HcDown => with_qwen4exp_hc_packed_projection_override(
            Qwen4ExpHcPackedProjectionArm::WideF32Down,
            SELECTED_START,
            prompt_tokens - SELECTED_START,
            || {
                let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
                f()
            },
        ),
    }
}

fn treatment_sha256(records: &[Qwen4ExpHcPackedProjectionRecord]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"qwen4exp-selected-quality-hc-treatment-records-v1\0");
    for record in records {
        digest.update(
            match record.arm {
                Qwen4ExpHcPackedProjectionArm::WideF32Down => 1_u32,
            }
            .to_le_bytes(),
        );
        for value in [
            record.start_position,
            record.tokens,
            record.n_in,
            record.n_out,
        ] {
            digest.update((value as u64).to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn validate_treatment(
    arm: LocalArm,
    prompt_tokens: usize,
    records: &[Qwen4ExpHcPackedProjectionRecord],
) {
    let expected = usize::from(arm == LocalArm::F32HcDown && prompt_tokens > SELECTED_START) * 96;
    assert_eq!(records.len(), expected, "{} treatment count", arm.label());
    assert!(records.iter().all(|record| {
        record.arm == Qwen4ExpHcPackedProjectionArm::WideF32Down
            && record.start_position == SELECTED_START
            && record.tokens == prompt_tokens - SELECTED_START
            && record.n_in == 10_240
            && record.n_out == 320
    }));
}

fn validate_prefill_topology(
    arm: LocalArm,
    prompt_tokens: usize,
    timing: Qwen4ExpPrefillTiming,
) -> Value {
    assert_eq!(timing.token_count, prompt_tokens);
    let selected = arm != LocalArm::DefaultSafe && prompt_tokens > SELECTED_START;
    let expected_packed = if selected {
        prompt_tokens
    } else {
        prompt_tokens.min(SELECTED_START)
    };
    let expected_commands = if prompt_tokens == SELECTED_START {
        2
    } else if selected {
        3
    } else {
        2 + prompt_tokens - SELECTED_START
    };
    assert_eq!(timing.packed_token_count, expected_packed);
    assert_eq!(timing.contains_selection, selected);
    assert_eq!(timing.command_count, expected_commands);
    json!({
        "token_count": timing.token_count,
        "packed_token_count": timing.packed_token_count,
        "contains_selection": timing.contains_selection,
        "command_count": timing.command_count,
        "encode_cpu_ms": timing.encode_cpu_ms,
        "completion_wait_ms": timing.completion_wait_ms,
        "gpu_ms": timing.complete_gpu_ms(),
        "gpu_samples": timing.gpu_samples,
        "total_wall_ms": timing.total_wall_ms,
        "outside_gpu_ms": timing.outside_gpu_ms(),
    })
}

fn attach_run_binding(
    observation: &mut Value,
    evidence_binding_sha256: &str,
    fixture_manifest_sha256: &str,
    fixture_id: &str,
    input_sha256: &str,
    operation: &Operation,
    identity: &ReplayIdentity,
) {
    let semantic_payload = serde_json::to_vec(observation).unwrap();
    let semantic_payload_sha256 = sha256_bytes(&semantic_payload);
    let domain = format!(
        concat!(
            "qwen4exp-selected-quality-run-binding-v1\0",
            "evidence_binding_sha256={}\n",
            "fixture_manifest_sha256={}\n",
            "fixture_id={}\n",
            "input_sha256={}\n",
            "operation_ordinal={}\nphase={}\nmode={}\narm={}\n",
            "replay_identity_sha256={}\n",
            "semantic_payload_sha256={}\n"
        ),
        evidence_binding_sha256,
        fixture_manifest_sha256,
        fixture_id,
        input_sha256,
        operation.ordinal,
        operation.phase,
        operation.mode,
        operation.arm,
        identity.binding_sha256(),
        semantic_payload_sha256,
    );
    observation.as_object_mut().unwrap().insert(
        "binding".into(),
        json!({
            "semantic_payload_encoding": "serde_json compact UTF-8 over the observation before its binding field",
            "semantic_payload_bytes": semantic_payload.len(),
            "semantic_payload_sha256": semantic_payload_sha256,
            "semantic_payload_json_compact": String::from_utf8(semantic_payload).unwrap(),
            "domain_utf8": domain,
            "sha256": sha256_bytes(domain.as_bytes()),
        }),
    );
}

fn record_run_binding(rows: &mut Vec<Value>, operation: &Operation, observation: &Value) {
    assert_eq!(operation.ordinal, rows.len());
    let binding_sha256 = observation["binding"]["sha256"].as_str().unwrap();
    assert!(is_lower_hex(binding_sha256, 64));
    rows.push(json!({
        "operation_ordinal": operation.ordinal,
        "fixture_id": operation.fixture_id,
        "phase": operation.phase,
        "mode": operation.mode,
        "arm": operation.arm,
        "run_binding_sha256": binding_sha256,
    }));
}

fn run_binding_root(rows: &[Value]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"qwen4exp-selected-quality-ordered-run-bindings-v1\0");
    for (ordinal, row) in rows.iter().enumerate() {
        assert_eq!(row["operation_ordinal"], ordinal);
        digest.update((ordinal as u64).to_le_bytes());
        digest.update(row["run_binding_sha256"].as_str().unwrap().as_bytes());
    }
    format!("{:x}", digest.finalize())
}

struct RunObservation {
    json: Value,
    identity: ReplayIdentity,
}

struct ScopeObservation {
    run: RunObservation,
    logits: Vec<f32>,
    state: Vec<Vec<u8>>,
    qsa_lengths: Vec<(u32, usize)>,
    ple_history: Vec<u32>,
}

fn replay_identity(
    endpoint: SnapshotIdentity,
    terminal: SnapshotIdentity,
    trace: LogitTrace,
    records: &[Qwen4ExpHcPackedProjectionRecord],
) -> ReplayIdentity {
    let (logit_trace_sha256, logit_trace_rows) = trace.finish();
    ReplayIdentity {
        endpoint,
        terminal,
        logit_trace_sha256,
        logit_trace_rows,
        treatment_sha256: treatment_sha256(records),
        treatment_records: records.len(),
    }
}

fn run_scope_operation(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    fixture: &ScopeFixture,
    prompt: &[u32],
    arm: LocalArm,
    operation: &Operation,
) -> ScopeObservation {
    reset_and_zero(runner);
    let ((logits, timing, endpoint, state, qsa_lengths, ple_history, trace), records) =
        with_local_arm(arm, prompt.len(), || {
            let logits = runner.prefill(prompt).unwrap().to_vec();
            let timing = runner.last_prefill_timing().unwrap();
            let endpoint = snapshot_identity(runner, &logits, "endpoint");
            let state = snapshot_persistent_state(runner);
            let qsa_lengths = runner.workspace.qsa_committed_lengths();
            let ple_history = runner.workspace.ple_prior_tokens().to_vec();
            let mut trace = LogitTrace::new();
            trace.push(&logits);
            (
                logits,
                timing,
                endpoint,
                state,
                qsa_lengths,
                ple_history,
                trace,
            )
        });
    validate_treatment(arm, prompt.len(), &records);
    assert_eq!(runner.next_position(), fixture.prompt_token_count);
    assert!(
        qsa_lengths
            .iter()
            .all(|(_, length)| *length == fixture.prompt_token_count)
    );
    let topology = validate_prefill_topology(arm, prompt.len(), timing);
    let identity = replay_identity(endpoint.clone(), endpoint, trace, &records);
    let observation = json!({
        "operation_ordinal": operation.ordinal,
        "fixture_id": fixture.fixture_id,
        "arm": arm.code(),
        "arm_label": arm.label(),
        "mode": operation.mode,
        "prefill": topology,
        "replay": identity.json(),
    });
    ScopeObservation {
        run: RunObservation {
            json: observation,
            identity,
        },
        logits,
        state,
        qsa_lengths,
        ple_history,
    }
}

fn run_natural_operation(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    fixture: &NaturalFixture,
    fixture_tokens: &[u32],
    arm: LocalArm,
    operation: &Operation,
) -> RunObservation {
    let (prompt, continuation) = fixture_tokens.split_at(fixture.prompt_token_count);
    assert_eq!(continuation.len(), CONTINUATION_TOKENS);
    reset_and_zero(runner);
    let ((timing, endpoint, terminal, trace, scored, nll_sum, top1_hits), records) =
        with_local_arm(arm, prompt.len(), || {
            let mut logits = runner.prefill(prompt).unwrap().to_vec();
            let timing = runner.last_prefill_timing().unwrap();
            let endpoint = snapshot_identity(runner, &logits, "endpoint");
            let mut trace = LogitTrace::new();
            let mut scored = Vec::with_capacity(continuation.len());
            let mut nll_sum = 0.0_f64;
            let mut top1_hits = 0_usize;
            for (ordinal, &target) in continuation.iter().enumerate() {
                trace.push(&logits);
                let row = score_token(&logits, ordinal, target);
                nll_sum += row.nll;
                top1_hits += usize::from(row.top1);
                scored.push(row.json);
                logits = runner.forward_token(target).unwrap().to_vec();
            }
            trace.push(&logits);
            let terminal = snapshot_identity(runner, &logits, "terminal");
            (
                timing, endpoint, terminal, trace, scored, nll_sum, top1_hits,
            )
        });
    validate_treatment(arm, prompt.len(), &records);
    assert_eq!(runner.next_position(), fixture_tokens.len());
    assert!(
        runner
            .workspace
            .qsa_committed_lengths()
            .iter()
            .all(|(_, length)| *length == fixture_tokens.len())
    );
    let topology = validate_prefill_topology(arm, prompt.len(), timing);
    let identity = replay_identity(endpoint, terminal, trace, &records);
    assert_eq!(identity.logit_trace_rows, CONTINUATION_TOKENS + 1);
    let observation = json!({
        "operation_ordinal": operation.ordinal,
        "fixture_id": fixture.fixture_id,
        "document_source_ordinal": fixture.document.source_ordinal,
        "prompt_token_count": fixture.prompt_token_count,
        "selected_suffix_tokens": fixture.selected_suffix_tokens,
        "arm": arm.code(),
        "arm_label": arm.label(),
        "mode": operation.mode,
        "prefill": topology,
        "continuation": {
            "tokens": continuation.len(),
            "nll_sum_f64": nll_sum,
            "mean_nll_f64": nll_sum / continuation.len() as f64,
            "top1_hits": top1_hits,
            "scored_rows": scored,
        },
        "replay": identity.json(),
    });
    RunObservation {
        json: observation,
        identity,
    }
}

fn run_open_greedy_operation(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    fixture: &NaturalFixture,
    prompt: &[u32],
    stop_tokens: &[u32],
    arm: LocalArm,
    operation: &Operation,
) -> RunObservation {
    reset_and_zero(runner);
    let ((timing, endpoint, terminal, trace, generated, fed_tokens, stopped), records) =
        with_local_arm(arm, prompt.len(), || {
            let mut logits = runner.prefill(prompt).unwrap().to_vec();
            let timing = runner.last_prefill_timing().unwrap();
            let endpoint = snapshot_identity(runner, &logits, "endpoint");
            let mut trace = LogitTrace::new();
            trace.push(&logits);
            let mut generated = Vec::new();
            let mut fed_tokens = 0_usize;
            let mut stopped = false;
            for _ in 0..32 {
                assert_eq!(logits.len(), VOCAB_SIZE);
                assert!(logits.iter().all(|value| value.is_finite()));
                let token = argmax(&logits) as u32;
                generated.push(token);
                if stop_tokens.contains(&token) {
                    stopped = true;
                    break;
                }
                logits = runner.forward_token(token).unwrap().to_vec();
                fed_tokens += 1;
                trace.push(&logits);
            }
            let terminal = snapshot_identity(runner, &logits, "terminal");
            (
                timing, endpoint, terminal, trace, generated, fed_tokens, stopped,
            )
        });
    validate_treatment(arm, prompt.len(), &records);
    assert_eq!(runner.next_position(), prompt.len() + fed_tokens);
    let topology = validate_prefill_topology(arm, prompt.len(), timing);
    let identity = replay_identity(endpoint, terminal, trace, &records);
    assert_eq!(identity.logit_trace_rows, fed_tokens + 1);
    let observation = json!({
        "operation_ordinal": operation.ordinal,
        "fixture_id": fixture.fixture_id,
        "prompt_token_count": fixture.prompt_token_count,
        "arm": arm.code(),
        "arm_label": arm.label(),
        "mode": operation.mode,
        "prefill": topology,
        "greedy": {
            "tie_policy": "lowest_token_id",
            "maximum_tokens": 32,
            "generated_token_ids": generated,
            "generated_token_ids_sha256_u32le": sha256_u32_le(
                b"qwen4exp-selected-quality-open-greedy-u32le-v1\0",
                &generated,
            ),
            "fed_non_stop_tokens": fed_tokens,
            "stopped_on_producer_token": stopped,
        },
        "replay": identity.json(),
    });
    RunObservation {
        json: observation,
        identity,
    }
}

fn run_retrieval_operation(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    fixture: &RetrievalFixture,
    prompt: &[u32],
    arm: LocalArm,
    operation: &Operation,
) -> RunObservation {
    reset_and_zero(runner);
    let ((timing, endpoint, terminal, trace, scored, nll_sum, greedy_prefix, exact_pass), records) =
        with_local_arm(arm, prompt.len(), || {
            let mut logits = runner.prefill(prompt).unwrap().to_vec();
            let timing = runner.last_prefill_timing().unwrap();
            let endpoint = snapshot_identity(runner, &logits, "endpoint");
            let mut trace = LogitTrace::new();
            let mut scored = Vec::with_capacity(fixture.answer_token_ids.len());
            let mut nll_sum = 0.0_f64;
            let mut greedy_prefix = Vec::new();
            let mut prefix_matches = true;
            for (ordinal, &target) in fixture.answer_token_ids.iter().enumerate() {
                trace.push(&logits);
                let prediction = argmax(&logits) as u32;
                if prefix_matches {
                    greedy_prefix.push(prediction);
                    prefix_matches = prediction == target;
                }
                let row = score_token(&logits, ordinal, target);
                nll_sum += row.nll;
                scored.push(row.json);
                logits = runner.forward_token(target).unwrap().to_vec();
            }
            trace.push(&logits);
            let exact_pass = if prefix_matches {
                let following = argmax(&logits) as u32;
                greedy_prefix.push(following);
                fixture.producer_stop_token_ids.contains(&following)
            } else {
                false
            };
            let terminal = snapshot_identity(runner, &logits, "terminal");
            (
                timing,
                endpoint,
                terminal,
                trace,
                scored,
                nll_sum,
                greedy_prefix,
                exact_pass,
            )
        });
    validate_treatment(arm, prompt.len(), &records);
    assert_eq!(
        runner.next_position(),
        prompt.len() + fixture.answer_token_ids.len()
    );
    assert!(greedy_prefix.len() <= fixture.answer_token_ids.len() + 1);
    assert!(greedy_prefix.len() <= fixture.max_generated_tokens);
    let topology = validate_prefill_topology(arm, prompt.len(), timing);
    let identity = replay_identity(endpoint, terminal, trace, &records);
    assert_eq!(
        identity.logit_trace_rows,
        fixture.answer_token_ids.len() + 1
    );
    let observation = json!({
        "operation_ordinal": operation.ordinal,
        "fixture_id": fixture.fixture_id,
        "kind": fixture.kind,
        "document_source_ordinal": fixture.document.source_ordinal,
        "prompt_token_count": prompt.len(),
        "selected_suffix_tokens": prompt.len() - SELECTED_START,
        "arm": arm.code(),
        "arm_label": arm.label(),
        "mode": operation.mode,
        "prefill": topology,
        "answer": {
            "expected_token_ids": fixture.answer_token_ids,
            "tokens": fixture.answer_token_ids.len(),
            "nll_sum_f64": nll_sum,
            "mean_nll_f64": nll_sum / fixture.answer_token_ids.len() as f64,
            "scored_rows": scored,
            "greedy_prefix_through_first_mismatch_or_stop": greedy_prefix,
            "greedy_prefix_contract": "true free-running prefix; after a mismatch, remaining expected answer tokens are teacher-forced only",
            "exact_pass": exact_pass,
        },
        "replay": identity.json(),
    });
    RunObservation {
        json: observation,
        identity,
    }
}

fn git_stdout(root: &Path, arguments: &[&str]) -> Vec<u8> {
    let output = std::process::Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

struct OutputReservation {
    lock_path: PathBuf,
    active: bool,
}

impl OutputReservation {
    fn acquire(output_path: &Path) -> Self {
        assert!(
            !output_path.exists(),
            "acquisition output must not already exist"
        );
        let parent = output_path.parent().expect("output path has no parent");
        assert!(parent.exists(), "output parent does not exist");
        let file_name = output_path.file_name().unwrap().to_string_lossy();
        let lock_path = parent.join(format!(".{file_name}.qwen4exp-selected-quality.lock"));
        let mut lock = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .expect("acquisition lock already exists");
        writeln!(
            lock,
            "pid={}\noutput={}",
            std::process::id(),
            output_path.display()
        )
        .unwrap();
        lock.sync_all().unwrap();
        std::fs::File::open(parent).unwrap().sync_all().unwrap();
        Self {
            lock_path,
            active: true,
        }
    }

    fn release(&mut self) -> std::io::Result<()> {
        if self.active {
            std::fs::remove_file(&self.lock_path)?;
            self.active = false;
            std::fs::File::open(self.lock_path.parent().unwrap())?.sync_all()?;
        }
        Ok(())
    }
}

impl Drop for OutputReservation {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

struct TemporaryReport {
    path: PathBuf,
    active: bool,
}

impl TemporaryReport {
    fn remove(&mut self) -> std::io::Result<()> {
        if self.active {
            std::fs::remove_file(&self.path)?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TemporaryReport {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}

struct FinalLinkRollback {
    path: PathBuf,
    active: bool,
}

impl FinalLinkRollback {
    fn rollback(&mut self) -> std::io::Result<()> {
        if self.active {
            std::fs::remove_file(&self.path)?;
            self.active = false;
            std::fs::File::open(self.path.parent().unwrap())?.sync_all()?;
        }
        Ok(())
    }

    fn commit(&mut self) {
        self.active = false;
    }
}

impl Drop for FinalLinkRollback {
    fn drop(&mut self) {
        let _ = self.rollback();
    }
}

fn publish_report_atomically(
    reservation: &mut OutputReservation,
    output_path: &Path,
    report_bytes: &[u8],
) -> String {
    assert!(
        !output_path.exists(),
        "final report path was claimed during acquisition"
    );
    let parent = output_path.parent().unwrap();
    let file_name = output_path.file_name().unwrap().to_string_lossy();
    let report_sha256 = sha256_bytes(report_bytes);
    let temporary_path = parent.join(format!(
        ".{file_name}.tmp.{}.{}",
        std::process::id(),
        &report_sha256[..16]
    ));
    let mut temporary = TemporaryReport {
        path: temporary_path.clone(),
        active: true,
    };
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .expect("temporary acquisition report already exists");
    output.write_all(report_bytes).unwrap();
    output.sync_all().unwrap();
    drop(output);
    std::fs::hard_link(&temporary_path, output_path)
        .expect("atomically publish acquisition report without clobbering");
    let mut final_link = FinalLinkRollback {
        path: output_path.to_path_buf(),
        active: true,
    };
    if let Err(sync_error) = std::fs::File::open(parent).and_then(|directory| directory.sync_all())
    {
        let rollback = final_link.rollback();
        panic!("final report directory sync failed: {sync_error}; rollback={rollback:?}");
    }
    final_link.commit();
    if let Err(error) = temporary.remove() {
        eprintln!("warning: published report but could not remove temporary file: {error}");
    }
    if let Err(error) = reservation.release() {
        eprintln!("warning: published report but could not remove reservation lock: {error}");
    }
    report_sha256
}

fn validate_arithmetic_policy_environment() -> Value {
    const ALLOWED_QWEN_ENV: [&str; 4] = [
        "QWEN4EXP_Q3_K_XL_RUNTIME_GGUF",
        "QWEN4EXP_SELECTED_QUALITY_OUT",
        "QWEN4EXP_SELECTED_QUALITY_SOURCE",
        "QWEN4EXP_SELECTED_QUALITY_DIFF_SHA256",
    ];
    let allowed = ALLOWED_QWEN_ENV.into_iter().collect::<BTreeSet<_>>();
    let unexpected = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| name.starts_with("QWEN") && !allowed.contains(name.as_str()))
        .collect::<Vec<_>>();
    assert!(
        unexpected.is_empty(),
        "undeclared QWEN arithmetic/environment overrides are forbidden: {unexpected:?}"
    );
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    json!({
        "undeclared_qwen_environment_overrides_rejected": true,
        "allowed_qwen_environment_names": ALLOWED_QWEN_ENV,
        "qwen4exp_moe_iq3_fast": {
            "effective": true,
            "authority": "cfg(test) thread-local override around all 78 local operations",
        },
        "qwen4exp_packed_router_e8p32_strict": {
            "effective": true,
            "authority": "cfg(test) thread-local override around all 78 local operations",
        },
        "qwen_matvec_q8_0_lcpp": {
            "effective": true,
            "authority": "default-on source policy after rejecting its environment override",
        },
        "all_other_kernel_policies": "source-bound defaults in the exact test executable and embedded metallib",
    })
}

fn kernel_source_manifest(root: &Path) -> Value {
    let kernels_dir = root.join("kernels");
    let mut paths = std::fs::read_dir(&kernels_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("metal"))
        .collect::<Vec<_>>();
    paths.sort();
    assert!(!paths.is_empty());
    let mut domain = String::from("qwen4exp-selected-quality-kernel-sources-v1\0");
    let mut rows = Vec::new();
    for path in paths {
        let relative = path.strip_prefix(root).unwrap();
        let relative_utf8 = relative.to_str().unwrap();
        let tracked = std::process::Command::new("git")
            .args(["ls-files", "--error-unmatch", "--", relative_utf8])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            tracked.status.success(),
            "untracked Metal source {relative_utf8}"
        );
        let bytes = std::fs::read(&path).unwrap();
        let content_sha256 = sha256_bytes(&bytes);
        domain.push_str(&format!(
            "{relative_utf8}\t{}\t{content_sha256}\n",
            bytes.len()
        ));
        rows.push(json!({
            "path": relative_utf8,
            "bytes": bytes.len(),
            "sha256": content_sha256,
        }));
    }
    json!({
        "schema": "qwen4exp-selected-quality-kernel-sources-v1",
        "discovery": "sorted direct children of workspace kernels/ with extension .metal, matching crates/qwen-llm/build.rs",
        "domain_utf8": domain,
        "sha256": sha256_bytes(domain.as_bytes()),
        "files": rows,
    })
}

fn validate_path_dependency(
    repository: &Path,
    expected_commit: &str,
    tree_spec: &str,
    expected_tree: &str,
    scopes: &[&str],
) -> Value {
    let commit = String::from_utf8(git_stdout(repository, &["rev-parse", "HEAD"]))
        .unwrap()
        .trim()
        .to_string();
    assert_eq!(commit, expected_commit);
    let tree = String::from_utf8(git_stdout(repository, &["rev-parse", tree_spec]))
        .unwrap()
        .trim()
        .to_string();
    assert_eq!(tree, expected_tree);
    let mut status_arguments = vec!["status", "--porcelain=v1", "--untracked-files=all", "--"];
    status_arguments.extend_from_slice(scopes);
    let scoped_status = git_stdout(repository, &status_arguments);
    assert!(
        scoped_status.is_empty(),
        "path dependency {} scopes {scopes:?} are dirty:\n{}",
        repository.display(),
        String::from_utf8_lossy(&scoped_status)
    );
    let full_status = git_stdout(
        repository,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    );
    json!({
        "repository_path": repository,
        "commit": commit,
        "tree_spec": tree_spec,
        "tree": tree,
        "clean_scopes": scopes,
        "scoped_status_sha256": sha256_bytes(&scoped_status),
        "full_status_bytes": full_status.len(),
        "full_status_sha256": sha256_bytes(&full_status),
        "qualification": "the pinned Git tree plus an empty scoped status identify every path-dependency source byte used by this workspace",
    })
}

fn validate_source_provenance(root: &Path, fixtures: &ValidatedFixtures) -> Value {
    let declared_source_commit = std::env::var("QWEN4EXP_SELECTED_QUALITY_SOURCE")
        .expect("QWEN4EXP_SELECTED_QUALITY_SOURCE must name the committed harness source");
    let declared_diff_sha256 = std::env::var("QWEN4EXP_SELECTED_QUALITY_DIFF_SHA256")
        .expect("QWEN4EXP_SELECTED_QUALITY_DIFF_SHA256 must identify the tracked diff");
    assert!(is_lower_hex(&declared_source_commit, 40));
    assert!(is_lower_hex(&declared_diff_sha256, 64));
    let actual_source_commit = String::from_utf8(git_stdout(root, &["rev-parse", "HEAD"]))
        .unwrap()
        .trim()
        .to_string();
    assert_eq!(actual_source_commit, declared_source_commit);
    let tracked_diff = git_stdout(root, &["diff", "--binary", "--no-ext-diff", "HEAD", "--"]);
    let actual_diff_sha256 = sha256_bytes(&tracked_diff);
    assert_eq!(actual_diff_sha256, declared_diff_sha256);
    assert_eq!(
        actual_diff_sha256, EMPTY_SHA256,
        "acquisition requires a clean tracked tree"
    );
    let kernels = kernel_source_manifest(root);
    let kernel_source_manifest_sha256 = kernels["sha256"].as_str().unwrap().to_string();
    let code_root = root.parent().unwrap();
    let path_dependencies = json!({
        "gguf_rs": validate_path_dependency(
            &code_root.join("gguf"),
            GGUF_RS_COMMIT,
            "HEAD^{tree}",
            GGUF_RS_TREE,
            &["."],
        ),
        "llama_cpp_sys_2": validate_path_dependency(
            &code_root.join("llama-cpp-rs"),
            LLAMA_CPP_RS_COMMIT,
            "HEAD:llama-cpp-sys-2",
            LLAMA_CPP_SYS_TREE,
            &["Cargo.toml", "llama-cpp-sys-2"],
        ),
    });
    let path_dependencies_compact = serde_json::to_vec(&path_dependencies).unwrap();
    let path_dependencies_sha256 = sha256_bytes(&path_dependencies_compact);
    let packet_dir = "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg";
    let mut required_paths = vec![
        "Cargo.toml".to_string(),
        "Cargo.lock".to_string(),
        "crates/qwen-llm/build.rs".to_string(),
        "crates/qwen-llm/src/qwen4exp_runtime.rs".to_string(),
        "crates/qwen-llm/src/qwen4exp_selected_quality.rs".to_string(),
        "crates/qwen-llm/examples/qwen4exp_selected_quality_prepare.rs".to_string(),
        "scripts/bench/qwen4exp_selected_quality_prepare.py".to_string(),
        "scripts/bench/qwen4exp_selected_quality_analyze.py".to_string(),
        "scripts/bench/qwen4exp_selected_quality_llama.py".to_string(),
        format!("{packet_dir}/README.md"),
        format!("{packet_dir}/fixtures.json"),
    ];
    required_paths.extend(
        fixtures
            .manifest
            .natural_fixtures
            .iter()
            .map(|fixture| format!("{packet_dir}/{}", fixture.tokens.path))
            .chain(std::iter::once(format!(
                "{packet_dir}/{}",
                fixtures.manifest.scope_control.tokens.path
            )))
            .chain(
                fixtures
                    .manifest
                    .retrieval_fixtures
                    .iter()
                    .map(|fixture| format!("{packet_dir}/{}", fixture.tokens.path)),
            ),
    );
    required_paths.sort();
    required_paths.dedup();
    for path in &required_paths {
        let output = std::process::Command::new("git")
            .args(["ls-files", "--error-unmatch", "--", path])
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "required acquisition input is not tracked: {path}"
        );
    }
    let scoped_status = git_stdout(
        root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates/qwen-llm",
            "kernels",
            "scripts/bench/qwen4exp_selected_quality_prepare.py",
            "scripts/bench/qwen4exp_selected_quality_analyze.py",
            "scripts/bench/qwen4exp_selected_quality_llama.py",
            packet_dir,
        ],
    );
    assert!(
        scoped_status.is_empty(),
        "acquisition source scope contains untracked or dirty paths:\n{}",
        String::from_utf8_lossy(&scoped_status)
    );
    let full_status = git_stdout(root, &["status", "--porcelain=v1", "--untracked-files=all"]);
    let required_head_path_count = required_paths.len();
    json!({
        "source_commit": actual_source_commit,
        "tracked_diff_definition": "sha256(git diff --binary --no-ext-diff HEAD --)",
        "tracked_diff_bytes": tracked_diff.len(),
        "tracked_diff_sha256": actual_diff_sha256,
        "required_head_paths": required_paths,
        "required_head_path_count": required_head_path_count,
        "scoped_status_bytes": scoped_status.len(),
        "scoped_status_sha256": sha256_bytes(&scoped_status),
        "full_worktree_status_bytes": full_status.len(),
        "full_worktree_status_sha256": sha256_bytes(&full_status),
        "kernel_source_manifest": kernels,
        "kernel_source_manifest_sha256": kernel_source_manifest_sha256,
        "path_dependencies": path_dependencies,
        "path_dependencies_json_compact": String::from_utf8(path_dependencies_compact).unwrap(),
        "path_dependencies_sha256": path_dependencies_sha256,
        "cleanliness_scope": "all tracked files plus untracked files under the crate, kernels, packet, preparation script, and workspace manifests; both external path dependencies have pinned Git trees and clean relevant scopes",
    })
}

fn retained_model_stamp_sha256(gguf: &GgufFile) -> String {
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let mut domain = String::from("qwen4exp-selected-quality-model-stamps-v1\0");
    for stamp in stamps {
        domain.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            stamp.shard_idx,
            stamp.device,
            stamp.inode,
            stamp.size,
            stamp.mtime_sec,
            stamp.mtime_nsec,
            stamp.ctime_sec,
            stamp.ctime_nsec,
        ));
    }
    sha256_bytes(domain.as_bytes())
}

fn validate_model_artifact(model_path: &Path, model: &ModelLock) -> (GgufFile, Value, String) {
    let gguf = GgufFile::open(model_path).expect("open released UD-Q3_K_XL GGUF");
    assert_eq!(gguf.shards.len(), model.shards.len());
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    assert_eq!(stamps.len(), model.shards.len());
    let mut actual_manifest_domain = String::from("qwen4exp-release-model-shard-manifest-v1\0");
    let mut stamp_domain = String::from("qwen4exp-selected-quality-model-stamps-v1\0");
    let mut rows = Vec::new();
    for (index, ((shard, stamp), expected)) in gguf
        .shards
        .iter()
        .zip(&stamps)
        .zip(&model.shards)
        .enumerate()
    {
        assert_eq!(expected.index, index);
        let file_name = shard.path.file_name().unwrap().to_str().unwrap();
        let bytes = shard.mmap_len() as u64;
        assert_eq!(file_name, expected.filename);
        assert_eq!(bytes, expected.bytes);
        assert_eq!(stamp.shard_idx, index);
        assert_eq!(stamp.path, shard.path);
        assert_eq!(stamp.size, expected.bytes);
        let content_sha256 = sha256_bytes(shard.mmap_bytes());
        assert_eq!(
            content_sha256, expected.sha256,
            "model shard {index} content"
        );
        actual_manifest_domain.push_str(&format!(
            "{index}\t{file_name}\t{bytes}\t{content_sha256}\n"
        ));
        stamp_domain.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            stamp.shard_idx,
            stamp.device,
            stamp.inode,
            stamp.size,
            stamp.mtime_sec,
            stamp.mtime_nsec,
            stamp.ctime_sec,
            stamp.ctime_nsec,
        ));
        rows.push(json!({
            "index": index,
            "path": &stamp.path,
            "file_name": file_name,
            "bytes": bytes,
            "sha256": content_sha256,
            "retained_descriptor_stamp": {
                "device": stamp.device,
                "inode": stamp.inode,
                "mtime_sec": stamp.mtime_sec,
                "mtime_nsec": stamp.mtime_nsec,
                "ctime_sec": stamp.ctime_sec,
                "ctime_nsec": stamp.ctime_nsec,
            },
        }));
    }
    assert_eq!(actual_manifest_domain, model.shard_manifest_domain_utf8);
    let actual_manifest_sha256 = sha256_bytes(actual_manifest_domain.as_bytes());
    assert_eq!(actual_manifest_sha256, model.shard_manifest_sha256);
    let stamp_sha256 = sha256_bytes(stamp_domain.as_bytes());
    assert_eq!(retained_model_stamp_sha256(&gguf), stamp_sha256);
    let config = Qwen4ExpConfig::from_gguf(&gguf).unwrap();
    assert_eq!(config, Qwen4ExpConfig::flash_next_reference());
    (
        gguf,
        json!({
            "repository": model.repository,
            "revision": model.revision,
            "quant": model.quant,
            "qualification": "every local shard was hashed in full before the first model call",
            "shard_manifest_domain_utf8": actual_manifest_domain,
            "shard_manifest_sha256": actual_manifest_sha256,
            "retained_descriptor_stamp_domain_utf8": stamp_domain,
            "retained_descriptor_stamp_sha256": stamp_sha256,
            "mapped_bytes_hashed_directly": true,
            "retained_stamps_revalidated_after_hashing": true,
            "shards": rows,
            "config_equals_flash_next_reference": true,
        }),
        stamp_sha256,
    )
}

fn validate_tokenizer_artifact(gguf: &GgufFile, tokenizer: &TokenizerLock) -> Value {
    let identity = qwen4exp_tokenizer_identity_sha256(gguf).unwrap();
    assert_eq!(identity, QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256);
    let identity_hex = identity
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(identity_hex, tokenizer.identity_sha256);
    let loaded = Tokenizer::from_gguf(gguf).unwrap();
    assert_eq!(loaded.n_vocab() as usize, tokenizer.vocab_size);
    let stop_tokens = gguf
        .stop_token_ids()
        .unwrap()
        .into_iter()
        .map(|token| u32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(stop_tokens, tokenizer.producer_stop_token_ids);
    assert_eq!(
        gguf.get_str("tokenizer.ggml.model").unwrap(),
        tokenizer.model
    );
    assert_eq!(
        gguf.get_str("tokenizer.ggml.pre").unwrap(),
        tokenizer.pretokenizer
    );
    let chat_template = gguf.get_str("tokenizer.chat_template").unwrap();
    assert_eq!(
        sha256_bytes(chat_template.as_bytes()),
        tokenizer.chat_template_sha256
    );
    json!({
        "identity_schema": tokenizer.identity_schema,
        "identity_sha256": identity_hex,
        "vocab_size": loaded.n_vocab(),
        "producer_stop_token_ids": tokenizer.producer_stop_token_ids,
        "model": tokenizer.model,
        "pretokenizer": tokenizer.pretokenizer,
        "chat_template_sha256": tokenizer.chat_template_sha256,
        "qualification_scope": "released-artifact evidence only; not a production admission rule",
    })
}

fn natural_fixture<'a>(manifest: &'a FixtureManifest, fixture_id: &str) -> &'a NaturalFixture {
    manifest
        .natural_fixtures
        .iter()
        .find(|fixture| fixture.fixture_id == fixture_id)
        .unwrap_or_else(|| panic!("unknown natural fixture {fixture_id}"))
}

fn retrieval_fixture<'a>(manifest: &'a FixtureManifest, fixture_id: &str) -> &'a RetrievalFixture {
    manifest
        .retrieval_fixtures
        .iter()
        .find(|fixture| fixture.fixture_id == fixture_id)
        .unwrap_or_else(|| panic!("unknown retrieval fixture {fixture_id}"))
}

fn operation_input_sha256<'a>(manifest: &'a FixtureManifest, operation: &Operation) -> &'a str {
    match operation.phase.as_str() {
        "scope_control" => &manifest.scope_control.tokens.sha256_raw_i32le,
        "natural_semantic" | "open_greedy" | "reverse_replay" => {
            &natural_fixture(manifest, &operation.fixture_id)
                .tokens
                .sha256_raw_i32le
        }
        "retrieval_semantic" => {
            &retrieval_fixture(manifest, &operation.fixture_id)
                .tokens
                .sha256_raw_i32le
        }
        _ => panic!("no local input for phase {}", operation.phase),
    }
}

#[test]
#[ignore = "set the released GGUF and QWEN4EXP_SELECTED_QUALITY_{OUT,SOURCE,DIFF_SHA256}"]
fn released_selected_quality_local_abc_acquisition() {
    let output_path = PathBuf::from(
        std::env::var_os("QWEN4EXP_SELECTED_QUALITY_OUT")
            .expect("QWEN4EXP_SELECTED_QUALITY_OUT must name a new JSON report"),
    );
    let mut output_reservation = OutputReservation::acquire(&output_path);
    let model_path = PathBuf::from(
        std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard"),
    );

    let arithmetic_policy = validate_arithmetic_policy_environment();
    let arithmetic_policy_compact = serde_json::to_vec(&arithmetic_policy).unwrap();
    let arithmetic_policy_sha256 = sha256_bytes(&arithmetic_policy_compact);
    let fixtures = validate_fixture_set();
    let root = repository_root();
    let source = validate_source_provenance(&root, &fixtures);
    let test_executable = std::env::current_exe().unwrap();
    let test_executable_bytes = std::fs::metadata(&test_executable).unwrap().len();
    let test_executable_sha256 = sha256_file(&test_executable);
    let scorer_source_sha256 = sha256_bytes(SCORER_SOURCE_BYTES);
    let metallib_sha256 = sha256_bytes(crate::KERNELS_METALLIB);
    let (gguf, model_report, model_stamp_sha256) =
        validate_model_artifact(&model_path, &fixtures.manifest.acquisition_model_lock);
    let tokenizer_report = validate_tokenizer_artifact(&gguf, &fixtures.manifest.tokenizer);

    let ctx = MetalContext::new().expect("initialize Metal");
    let max_forward_tokens = fixtures
        .manifest
        .natural_fixtures
        .iter()
        .map(|fixture| fixture.prompt_token_count + CONTINUATION_TOKENS)
        .max()
        .unwrap();
    assert_eq!(max_forward_tokens, 4_195);
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
        &Qwen4ExpConfig::flash_next_reference(),
        max_forward_tokens,
    )
    .unwrap();
    assert_eq!(retained_model_stamp_sha256(&gguf), model_stamp_sha256);
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, 4_099).unwrap();
    assert_eq!(loaded.packed_prefill_capacity(), Some(2_048));
    assert!(loaded.packed_selected_capable());
    let admission = loaded.admission();
    let observed_weight_bytes = loaded.observed_weight_bytes();
    let observed_session_bytes = loaded.observed_session_bytes();
    let mut runner = loaded.create_runner(&ctx).unwrap();

    let evidence_domain = format!(
        concat!(
            "qwen4exp-selected-quality-local-abc-evidence-v1\0",
            "packet_id={}\n",
            "fixture_manifest_sha256={}\n",
            "source_commit={}\ntracked_diff_sha256={}\n",
            "kernel_source_manifest_sha256={}\n",
            "path_dependencies_sha256={}\n",
            "test_executable_sha256={}\n",
            "scorer_source_sha256={}\n",
            "embedded_metallib_sha256={}\n",
            "arithmetic_policy_sha256={}\n",
            "model_shard_manifest_sha256={}\n",
            "model_retained_stamps_sha256={}\n",
            "device_name={}\ndevice_registry_id={}\n"
        ),
        PACKET_ID,
        FIXTURE_MANIFEST_SHA256,
        source["source_commit"].as_str().unwrap(),
        source["tracked_diff_sha256"].as_str().unwrap(),
        source["kernel_source_manifest_sha256"].as_str().unwrap(),
        source["path_dependencies_sha256"].as_str().unwrap(),
        test_executable_sha256,
        scorer_source_sha256,
        metallib_sha256,
        arithmetic_policy_sha256,
        fixtures
            .manifest
            .acquisition_model_lock
            .shard_manifest_sha256,
        model_stamp_sha256,
        ctx.device.name(),
        ctx.device.registryID(),
    );
    let evidence_binding_sha256 = sha256_bytes(evidence_domain.as_bytes());

    let mut scope_rows = Vec::new();
    let mut natural_rows = Vec::new();
    let mut open_greedy_rows = Vec::new();
    let mut retrieval_rows = Vec::new();
    let mut reverse_rows = Vec::new();
    let mut ordered_run_bindings = Vec::new();
    let mut scope_baseline: Option<ScopeObservation> = None;
    let mut natural_identities = BTreeMap::<(String, LocalArm), ReplayIdentity>::new();
    with_qwen4exp_moe_iq3_fast_override(true, || {
        with_qwen4exp_packed_router_e8p32_strict_override(true, || {
            for operation in &fixtures.manifest.execution.operation_plan[..LOCAL_OPERATION_COUNT] {
                let arm = LocalArm::from_code(&operation.arm);
                let input_sha256 = operation_input_sha256(&fixtures.manifest, operation);
                match operation.phase.as_str() {
                    "scope_control" => {
                        let mut observed = run_scope_operation(
                            &mut runner,
                            &fixtures.manifest.scope_control,
                            &fixtures.tokens[&operation.fixture_id],
                            arm,
                            operation,
                        );
                        if let Some(baseline) = &scope_baseline {
                            assert_f32_bits_eq(
                                "scope endpoint logits",
                                &baseline.logits,
                                &observed.logits,
                            );
                            assert_state_bytes_eq(
                                "scope persistent state",
                                &baseline.state,
                                &observed.state,
                            );
                            assert_eq!(baseline.qsa_lengths, observed.qsa_lengths);
                            assert_eq!(baseline.ple_history, observed.ple_history);
                            assert_eq!(baseline.run.identity, observed.run.identity);
                        } else {
                            assert_eq!(arm, LocalArm::DefaultSafe);
                        }
                        attach_run_binding(
                            &mut observed.run.json,
                            &evidence_binding_sha256,
                            FIXTURE_MANIFEST_SHA256,
                            &operation.fixture_id,
                            input_sha256,
                            operation,
                            &observed.run.identity,
                        );
                        record_run_binding(
                            &mut ordered_run_bindings,
                            operation,
                            &observed.run.json,
                        );
                        scope_rows.push(observed.run.json.clone());
                        if scope_baseline.is_none() {
                            scope_baseline = Some(observed);
                        }
                    }
                    "natural_semantic" => {
                        let fixture = natural_fixture(&fixtures.manifest, &operation.fixture_id);
                        let mut observed = run_natural_operation(
                            &mut runner,
                            fixture,
                            &fixtures.tokens[&operation.fixture_id],
                            arm,
                            operation,
                        );
                        assert!(
                            natural_identities
                                .insert(
                                    (operation.fixture_id.clone(), arm),
                                    observed.identity.clone()
                                )
                                .is_none()
                        );
                        attach_run_binding(
                            &mut observed.json,
                            &evidence_binding_sha256,
                            FIXTURE_MANIFEST_SHA256,
                            &operation.fixture_id,
                            input_sha256,
                            operation,
                            &observed.identity,
                        );
                        record_run_binding(&mut ordered_run_bindings, operation, &observed.json);
                        natural_rows.push(observed.json);
                    }
                    "open_greedy" => {
                        let fixture = natural_fixture(&fixtures.manifest, &operation.fixture_id);
                        let prompt =
                            &fixtures.tokens[&operation.fixture_id][..fixture.prompt_token_count];
                        let mut observed = run_open_greedy_operation(
                            &mut runner,
                            fixture,
                            prompt,
                            &fixtures.manifest.tokenizer.producer_stop_token_ids,
                            arm,
                            operation,
                        );
                        assert_eq!(
                            observed.identity.endpoint,
                            natural_identities[&(operation.fixture_id.clone(), arm)].endpoint,
                            "open-greedy prefill must replay its semantic endpoint"
                        );
                        attach_run_binding(
                            &mut observed.json,
                            &evidence_binding_sha256,
                            FIXTURE_MANIFEST_SHA256,
                            &operation.fixture_id,
                            input_sha256,
                            operation,
                            &observed.identity,
                        );
                        record_run_binding(&mut ordered_run_bindings, operation, &observed.json);
                        open_greedy_rows.push(observed.json);
                    }
                    "retrieval_semantic" => {
                        let fixture = retrieval_fixture(&fixtures.manifest, &operation.fixture_id);
                        let mut observed = run_retrieval_operation(
                            &mut runner,
                            fixture,
                            &fixtures.tokens[&operation.fixture_id],
                            arm,
                            operation,
                        );
                        attach_run_binding(
                            &mut observed.json,
                            &evidence_binding_sha256,
                            FIXTURE_MANIFEST_SHA256,
                            &operation.fixture_id,
                            input_sha256,
                            operation,
                            &observed.identity,
                        );
                        record_run_binding(&mut ordered_run_bindings, operation, &observed.json);
                        retrieval_rows.push(observed.json);
                    }
                    "reverse_replay" => {
                        let fixture = natural_fixture(&fixtures.manifest, &operation.fixture_id);
                        let mut observed = run_natural_operation(
                            &mut runner,
                            fixture,
                            &fixtures.tokens[&operation.fixture_id],
                            arm,
                            operation,
                        );
                        assert_eq!(
                            observed.identity,
                            natural_identities[&(operation.fixture_id.clone(), arm)],
                            "reverse replay must preserve every full-logit and state digest"
                        );
                        observed
                            .json
                            .as_object_mut()
                            .unwrap()
                            .insert("matches_initial_semantic_replay".into(), Value::Bool(true));
                        attach_run_binding(
                            &mut observed.json,
                            &evidence_binding_sha256,
                            FIXTURE_MANIFEST_SHA256,
                            &operation.fixture_id,
                            input_sha256,
                            operation,
                            &observed.identity,
                        );
                        record_run_binding(&mut ordered_run_bindings, operation, &observed.json);
                        reverse_rows.push(observed.json);
                    }
                    phase => panic!("unexpected local operation phase {phase}"),
                }
            }
        })
    });
    assert_eq!(scope_rows.len(), 3);
    assert_eq!(natural_rows.len(), 36);
    assert_eq!(open_greedy_rows.len(), 12);
    assert_eq!(retrieval_rows.len(), 24);
    assert_eq!(reverse_rows.len(), 3);
    assert_eq!(natural_identities.len(), 36);
    assert_eq!(ordered_run_bindings.len(), LOCAL_OPERATION_COUNT);
    let ordered_run_binding_root_sha256 = run_binding_root(&ordered_run_bindings);
    assert_eq!(retained_model_stamp_sha256(&gguf), model_stamp_sha256);

    let report = json!({
        "schema": "qwen4exp-selected-quality-local-abc-evidence",
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "local_abc_acquired_unanalyzed",
        "disposition": Value::Null,
        "fixture_manifest": {
            "path": "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/fixtures.json",
            "bytes": FIXTURE_MANIFEST_BYTES.len(),
            "sha256": FIXTURE_MANIFEST_SHA256,
        },
        "implementation": {
            "source": source,
            "test_executable": {
                "path": test_executable,
                "bytes": test_executable_bytes,
                "sha256": test_executable_sha256,
            },
            "scorer_source": {
                "path": "crates/qwen-llm/src/qwen4exp_selected_quality.rs",
                "bytes": SCORER_SOURCE_BYTES.len(),
                "sha256": scorer_source_sha256,
            },
            "embedded_metallib": {
                "bytes": crate::KERNELS_METALLIB.len(),
                "sha256": metallib_sha256,
            },
            "arithmetic_policy": arithmetic_policy,
            "arithmetic_policy_json_compact": String::from_utf8(arithmetic_policy_compact).unwrap(),
            "arithmetic_policy_sha256": arithmetic_policy_sha256,
            "evidence_domain_utf8": evidence_domain,
            "evidence_binding_sha256": evidence_binding_sha256,
            "ordered_run_bindings": ordered_run_bindings,
            "ordered_run_binding_root_sha256": ordered_run_binding_root_sha256,
        },
        "model": model_report,
        "tokenizer": tokenizer_report,
        "device": {
            "name": ctx.device.name().to_string(),
            "registry_id": ctx.device.registryID(),
            "max_threadgroup_memory_bytes": ctx.device.maxThreadgroupMemoryLength(),
        },
        "runtime": {
            "forward_limit": capacity.forward_limit(),
            "qsa_physical_capacity": capacity.qsa_physical_capacity(),
            "observed_weight_bytes": observed_weight_bytes,
            "observed_session_bytes": observed_session_bytes,
            "admission": {
                "aggregate_admitted": admission.aggregate.admitted,
                "weights_admitted": admission.weights.admitted,
                "session_admitted": admission.session.admitted,
            },
        },
        "scoring_contract": {
            "vocab_size": VOCAB_SIZE,
            "logits": "all F32 values must be finite",
            "nll": "max-subtracted F64 logsumexp over the complete row",
            "argmax_tie_policy": "lowest token ID",
            "natural_forwards": "score each of 96 current rows, then feed that target exactly once; terminal row is unscored",
            "retrieval_exact": "answer from generated token zero followed immediately by a producer stop",
        },
        "observations": {
            "scope_control": scope_rows,
            "natural_semantic": natural_rows,
            "open_greedy": open_greedy_rows,
            "retrieval_semantic": retrieval_rows,
            "reverse_replay": reverse_rows,
        },
    });
    let mut report_bytes = serde_json::to_vec_pretty(&report).unwrap();
    report_bytes.push(b'\n');
    let report_sha256 =
        publish_report_atomically(&mut output_reservation, &output_path, &report_bytes);
    eprintln!(
        "wrote Qwen4Exp selected-quality local A/B/C evidence to {} bytes={} sha256={report_sha256}",
        output_path.display(),
        report_bytes.len(),
    );
}
