//! Generate frozen token fixtures for the Flash-Next selected-quality packet.

use anyhow::{Context, Result, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::tokenizer::{
    QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256, Tokenizer, qwen4exp_tokenizer_identity_sha256,
    token_ids_sha256_i32le,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "qwen4exp-selected-quality-fixtures";
const SCHEMA_VERSION: u64 = 2;
const SOURCE_SCHEMA: &str = "qwen4exp-selected-quality-source-plan";
const PROMPT_SHAPES: [usize; 12] = [
    2_179, 2_179, 2_179, 2_563, 2_563, 2_563, 3_075, 3_075, 3_075, 4_099, 4_099, 4_099,
];
const ARM_ORDERS: [&str; 12] = [
    "ABC", "BCA", "CAB", "ACB", "CBA", "BAC", "ABC", "BCA", "CAB", "ACB", "CBA", "BAC",
];
const RETRIEVAL_ARM_ORDERS: [&str; 8] = ["ABC", "BCA", "CAB", "ACB", "CBA", "BAC", "ABC", "BCA"];
const CONTINUATION_TOKENS: usize = 96;
const SCOPE_TOKENS: usize = 2_051;
const RETRIEVAL_TOKENS: usize = 4_099;
const MAX_REQUIRED_DOCUMENT_TOKENS: usize = RETRIEVAL_TOKENS + CONTINUATION_TOKENS;
const MODEL_REPOSITORY: &str = "unsloth/Qwen3.8-Flash-Next-GGUF";
const MODEL_REVISION: &str = "8bdc666649440e9bdc97e16f3f75782c98478ff5";
const MODEL_SHARDS: [(&str, u64, &str); 3] = [
    (
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        10_946_624,
        "f2ef4328929d8b8c8930e2856eef52128dd4ce3425302f04bc3c657431cc4c49",
    ),
    (
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        49_983_253_824,
        "7d230e7c9421d868b89eebaf23033af0ea1a4e046956df00fb156814fb62346e",
    ),
    (
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
        39_992_153_376,
        "21d4f90f9cd7b7c3a1582667c20cb22f7b03de895b88a23bb20aaeaa44f2c199",
    ),
];
const BOOTSTRAP_RNG: &str = "splitmix64-rejection-u64-v1";
const BOOTSTRAP_SEED: u64 = 0x38f1_a9c5_d204_7e61;
const LLAMA_CPP_COMMIT: &str = "6c84c7d5d8833c6e0df69628f75a0f599797934e";

#[derive(Deserialize)]
struct SourcePlan {
    schema: String,
    schema_version: u64,
    source: Value,
    selection: Value,
    documents: Vec<SourceDocument>,
}

#[derive(Deserialize)]
struct SourceDocument {
    source_ordinal: usize,
    title: String,
    row_start: usize,
    row_end_exclusive: usize,
    utf8_bytes: usize,
    utf8_sha256: String,
    text: String,
}

struct TokenizedDocument {
    source_ordinal: usize,
    title: String,
    row_start: usize,
    row_end_exclusive: usize,
    utf8_bytes: usize,
    utf8_sha256: String,
    text: String,
    tokens: Vec<i32>,
}

#[derive(Clone, Copy)]
enum RetrievalKind {
    Single { position: usize },
    Double { first: usize, second: usize },
}

struct RetrievalSpec {
    fixture_id: &'static str,
    target: RetrievalKind,
    decoy: RetrievalKind,
    answer_tokens: usize,
}

struct RetrievalInsertion {
    position: usize,
    branch: &'static str,
    role: &'static str,
    payload: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn token_bytes(tokens: &[i32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect()
}

fn token_five_grams(tokens: &[i32]) -> BTreeSet<[i32; 5]> {
    tokens
        .windows(5)
        .map(|window| window.try_into().unwrap())
        .collect()
}

fn domain_token_sha256(fixture_id: &str, role: &str, tokens: &[i32]) -> (String, String) {
    let domain = format!(
        "qwen4exp-selected-quality-token-ids-i32le-v1\0fixture_id={fixture_id}\nrole={role}\ncount={}\n",
        tokens.len()
    );
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update(token_bytes(tokens));
    (domain, format!("{:x}", digest.finalize()))
}

fn output_file(path: &Path, bytes: &[u8], check: bool) -> Result<()> {
    let regenerate = matches!(
        std::env::var("QWEN4EXP_SELECTED_QUALITY_REGENERATE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    );
    if path.exists() {
        let existing = std::fs::read(path)
            .with_context(|| format!("read generated fixture {}", path.display()))?;
        if existing == bytes {
            return Ok(());
        }
        ensure!(
            !check && regenerate,
            "generated fixture drift: {}",
            path.display()
        );
    }
    ensure!(!check, "missing generated fixture: {}", path.display());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create fixture directory {}", parent.display()))?;
    }
    std::fs::write(path, bytes).with_context(|| format!("write fixture {}", path.display()))
}

fn fixture_token_record(
    fixture_id: &str,
    role: &str,
    relative_path: &str,
    tokens: &[i32],
) -> Value {
    let bytes = token_bytes(tokens);
    let (domain, domain_sha256) = domain_token_sha256(fixture_id, role, tokens);
    json!({
        "path": relative_path,
        "dtype": "i32",
        "byte_order": "little",
        "token_count": tokens.len(),
        "bytes": bytes.len(),
        "sha256": sha256(&bytes),
        "sha256_raw_i32le": token_ids_sha256_i32le(tokens),
        "digest_domain_utf8": domain,
        "sha256_domain_i32le": domain_sha256,
    })
}

fn document_record(document: &TokenizedDocument) -> Value {
    json!({
        "source_ordinal": document.source_ordinal,
        "title": document.title,
        "title_sha256_utf8": sha256(document.title.as_bytes()),
        "row_start": document.row_start,
        "row_end_exclusive": document.row_end_exclusive,
        "utf8_bytes": document.utf8_bytes,
        "utf8_sha256": document.utf8_sha256,
        "full_token_count": document.tokens.len(),
    })
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    tokenizer
        .encode(text, false)
        .with_context(|| format!("tokenize {} UTF-8 bytes", text.len()))
}

fn next_answer(
    tokenizer: &Tokenizer,
    desired_tokens: usize,
    used: &mut BTreeSet<Vec<i32>>,
    forbidden_texts: &[&str],
) -> Result<(String, Vec<i32>)> {
    const PREFIXES: &[u8] = b"QZXVJKRTPMNHFWYB";
    for width in 1_usize..=3 {
        for &prefix in PREFIXES {
            for ordinal in 0..16_u32.pow(width as u32) {
                let candidate = format!("{}{:0width$X}", char::from(prefix), ordinal);
                if forbidden_texts.iter().any(|text| text.contains(&candidate)) {
                    continue;
                }
                let tokens = encode(tokenizer, &candidate)?;
                if tokens.len() == desired_tokens && used.insert(tokens.clone()) {
                    return Ok((candidate, tokens));
                }
            }
        }
    }
    bail!("no unused opaque {desired_tokens}-token retrieval answer candidate")
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[i32]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();
    for &token in tokens {
        decoded.extend_from_slice(
            tokenizer
                .try_decode_piece_bytes_exact(token)
                .with_context(|| format!("decode frozen token {token}"))?,
        );
    }
    Ok(decoded)
}

fn count_bytes(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || needle.len() > haystack.len() {
        return 0;
    }
    haystack
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

fn push_arm_operations(
    operations: &mut Vec<Value>,
    phase: &str,
    fixture_id: &str,
    mode: &str,
    order: &str,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for arm in order.chars() {
        ensure!(matches!(arm, 'A' | 'B' | 'C'), "invalid local arm {arm}");
        ensure!(seen.insert(arm), "duplicate local arm {arm}");
        operations.push(json!({
            "ordinal": operations.len(),
            "phase": phase,
            "fixture_id": fixture_id,
            "mode": mode,
            "arm": arm.to_string(),
        }));
    }
    ensure!(seen.len() == 3, "local operation order must cover A/B/C");
    Ok(())
}

fn append_filler_until(
    prompt: &mut Vec<i32>,
    target: usize,
    filler: &[i32],
    cursor: &mut usize,
) -> Result<()> {
    ensure!(
        prompt.len() <= target,
        "retrieval payload crossed target {target}"
    );
    ensure!(!filler.is_empty(), "retrieval filler is empty");
    while prompt.len() < target {
        prompt.push(filler[*cursor % filler.len()]);
        *cursor += 1;
    }
    Ok(())
}

fn build_retrieval(
    tokenizer: &Tokenizer,
    document: &TokenizedDocument,
    spec: &RetrievalSpec,
    answer: &str,
    answer_tokens: &[i32],
    decoy_answer: &str,
    decoy_answer_tokens: &[i32],
    stop_tokens: &[i32],
) -> Result<(Vec<i32>, Value)> {
    let task_ordinal = document.source_ordinal;
    let key = format!("KEY{task_ordinal:03}Q");
    let relay = format!("RELAY{task_ordinal:03}M");
    let decoy_key = format!("KEY{task_ordinal:03}D");
    let decoy_relay = format!("RELAY{task_ordinal:03}N");
    let header = concat!(
        "<|im_start|>system\n",
        "You are a precise retrieval engine. Follow the user's records and output only the requested value.",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "Read the document and retain every explicit REFERENCE RECORD.\n\n"
    );
    let tail = format!(
        "\n\nQuestion: Follow the REFERENCE RECORD chain for {key}. What is its exact final value? Reply with only the value, with no punctuation or explanation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let mut prompt = encode(tokenizer, header)?;
    let tail_tokens = encode(tokenizer, &tail)?;
    let mut cursor = (task_ordinal * 977) % document.tokens.len();
    let mut insertions = Vec::new();
    let mut evidence = Vec::new();

    match spec.target {
        RetrievalKind::Single { position } => {
            insertions.push(RetrievalInsertion {
                position,
                branch: "target",
                role: "single_hop_value",
                payload: format!("\n\nREFERENCE RECORD: {key} HAS FINAL VALUE {answer}.\n\n"),
            });
        }
        RetrievalKind::Double { first, second } => {
            insertions.push(RetrievalInsertion {
                position: first,
                branch: "target",
                role: "two_hop_key_to_relay",
                payload: format!("\n\nREFERENCE RECORD: {key} POINTS TO {relay}.\n\n"),
            });
            insertions.push(RetrievalInsertion {
                position: second,
                branch: "target",
                role: "two_hop_relay_to_value",
                payload: format!("\n\nREFERENCE RECORD: {relay} HAS FINAL VALUE {answer}.\n\n"),
            });
        }
    }
    match spec.decoy {
        RetrievalKind::Single { position } => {
            insertions.push(RetrievalInsertion {
                position,
                branch: "decoy",
                role: "single_hop_value",
                payload: format!(
                    "\n\nREFERENCE RECORD: {decoy_key} HAS FINAL VALUE {decoy_answer}.\n\n"
                ),
            });
        }
        RetrievalKind::Double { first, second } => {
            insertions.push(RetrievalInsertion {
                position: first,
                branch: "decoy",
                role: "two_hop_key_to_relay",
                payload: format!("\n\nREFERENCE RECORD: {decoy_key} POINTS TO {decoy_relay}.\n\n"),
            });
            insertions.push(RetrievalInsertion {
                position: second,
                branch: "decoy",
                role: "two_hop_relay_to_value",
                payload: format!(
                    "\n\nREFERENCE RECORD: {decoy_relay} HAS FINAL VALUE {decoy_answer}.\n\n"
                ),
            });
        }
    }
    insertions.sort_by_key(|insertion| insertion.position);
    ensure!(
        insertions
            .windows(2)
            .all(|pair| pair[0].position < pair[1].position),
        "retrieval insertion positions must be unique"
    );
    let mut generated_fragments = vec![header.to_string(), tail.clone()];
    for insertion in insertions {
        append_filler_until(
            &mut prompt,
            insertion.position,
            &document.tokens,
            &mut cursor,
        )?;
        let payload_tokens = encode(tokenizer, &insertion.payload)?;
        let start = prompt.len();
        prompt.extend_from_slice(&payload_tokens);
        evidence.push(json!({
            "branch": insertion.branch,
            "role": insertion.role,
            "target_token_index": insertion.position,
            "realized_token_span": [start, prompt.len()],
            "payload_sha256_utf8": sha256(insertion.payload.as_bytes()),
        }));
        generated_fragments.push(insertion.payload);
    }

    let filler_end = RETRIEVAL_TOKENS
        .checked_sub(tail_tokens.len())
        .context("retrieval tail exceeds prompt")?;
    append_filler_until(&mut prompt, filler_end, &document.tokens, &mut cursor)?;
    prompt.extend_from_slice(&tail_tokens);
    ensure!(
        prompt.len() == RETRIEVAL_TOKENS,
        "retrieval prompt length drift"
    );

    let prompt_bytes = decode_tokens(tokenizer, &prompt)?;
    let symbols = [
        ("target_key", key.as_str()),
        ("target_relay", relay.as_str()),
        ("target_answer", answer),
        ("decoy_key", decoy_key.as_str()),
        ("decoy_relay", decoy_relay.as_str()),
        ("decoy_answer", decoy_answer),
    ];
    let symbol_audit = symbols
        .into_iter()
        .map(|(role, symbol)| {
            let source_occurrences = document.text.matches(symbol).count();
            let expected_prompt_occurrences = generated_fragments
                .iter()
                .map(|fragment| fragment.matches(symbol).count())
                .sum::<usize>();
            let actual_prompt_occurrences = count_bytes(&prompt_bytes, symbol.as_bytes());
            ensure!(
                source_occurrences == 0,
                "{role} occurs in retrieval filler source"
            );
            ensure!(
                actual_prompt_occurrences == expected_prompt_occurrences,
                "{role} occurrence count drift"
            );
            Ok(json!({
                "role": role,
                "utf8_sha256": sha256(symbol.as_bytes()),
                "source_document_occurrences": source_occurrences,
                "expected_prompt_occurrences": expected_prompt_occurrences,
                "actual_prompt_occurrences": actual_prompt_occurrences,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok((
        prompt,
        json!({
            "fixture_id": spec.fixture_id,
            "kind": match spec.target {
                RetrievalKind::Single { .. } => "single_hop",
                RetrievalKind::Double { .. } => "two_hop",
            },
            "document": document_record(document),
            "key": key,
            "relay": match spec.target {
                RetrievalKind::Single { .. } => Value::Null,
                RetrievalKind::Double { .. } => Value::String(relay),
            },
            "evidence": evidence,
            "symbol_exclusion_audit": symbol_audit,
            "answer_text": answer,
            "answer_text_sha256_utf8": sha256(answer.as_bytes()),
            "answer_token_ids": answer_tokens,
            "answer_token_count": answer_tokens.len(),
            "answer_token_ids_sha256_raw_i32le": token_ids_sha256_i32le(answer_tokens),
            "decoy": {
                "key": decoy_key,
                "relay": match spec.decoy {
                    RetrievalKind::Single { .. } => Value::Null,
                    RetrievalKind::Double { .. } => Value::String(decoy_relay),
                },
                "answer_text": decoy_answer,
                "answer_text_sha256_utf8": sha256(decoy_answer.as_bytes()),
                "answer_token_ids": decoy_answer_tokens,
                "answer_token_count": decoy_answer_tokens.len(),
                "answer_token_ids_sha256_raw_i32le": token_ids_sha256_i32le(decoy_answer_tokens),
            },
            "producer_stop_token_ids": stop_tokens,
            "max_generated_tokens": 8,
            "exact_match": {
                "normalization": "none",
                "require_answer_from_generated_token_zero": true,
                "require_complete_answer": true,
                "require_immediate_following_producer_stop": true,
            },
        }),
    ))
}

fn main() -> Result<()> {
    let model_path = PathBuf::from(
        std::env::var_os("QWEN4EXP_SELECTED_QUALITY_GGUF")
            .context("QWEN4EXP_SELECTED_QUALITY_GGUF")?,
    );
    let source_plan_path = PathBuf::from(
        std::env::var_os("QWEN4EXP_SELECTED_QUALITY_SOURCE_PLAN")
            .context("QWEN4EXP_SELECTED_QUALITY_SOURCE_PLAN")?,
    );
    let fixture_dir = PathBuf::from(
        std::env::var_os("QWEN4EXP_SELECTED_QUALITY_FIXTURE_DIR")
            .context("QWEN4EXP_SELECTED_QUALITY_FIXTURE_DIR")?,
    );
    let check = matches!(
        std::env::var("QWEN4EXP_SELECTED_QUALITY_CHECK").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    );

    let source_plan_bytes = std::fs::read(&source_plan_path)
        .with_context(|| format!("read source plan {}", source_plan_path.display()))?;
    let mut source_plan: SourcePlan =
        serde_json::from_slice(&source_plan_bytes).context("parse source plan")?;
    ensure!(
        source_plan.schema == SOURCE_SCHEMA,
        "source plan schema mismatch"
    );
    ensure!(
        source_plan.schema_version == 1,
        "source plan version mismatch"
    );

    let gguf = GgufFile::open(&model_path).context("open released GGUF")?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load released tokenizer")?;
    let tokenizer_identity =
        qwen4exp_tokenizer_identity_sha256(&gguf).context("fingerprint tokenizer")?;
    ensure!(
        tokenizer_identity == QWEN4EXP_RELEASE_TOKENIZER_IDENTITY_SHA256,
        "fixture generation requires the released tokenizer identity"
    );
    let stop_tokens = gguf.stop_token_ids().context("read producer stop tokens")?;

    let mut tokenized = Vec::new();
    for document in source_plan.documents {
        let tokens = encode(&tokenizer, &document.text)?;
        if tokens.len() >= MAX_REQUIRED_DOCUMENT_TOKENS {
            tokenized.push(TokenizedDocument {
                source_ordinal: document.source_ordinal,
                title: document.title,
                row_start: document.row_start,
                row_end_exclusive: document.row_end_exclusive,
                utf8_bytes: document.utf8_bytes,
                utf8_sha256: document.utf8_sha256,
                text: document.text,
                tokens,
            });
        }
    }
    ensure!(
        tokenized.len() >= 21,
        "fewer than 21 token-qualified documents"
    );
    let token_qualified_documents = tokenized.len();
    tokenized.truncate(21);

    let tokens_dir = fixture_dir.join("tokens");
    let mut natural = Vec::new();
    let mut expected_paths = BTreeSet::new();
    let mut fixture_tokens_for_exclusion = Vec::<(String, Vec<i32>)>::new();
    for index in 0..12 {
        let shape = PROMPT_SHAPES[index];
        let document = &tokenized[index];
        let fixture_id = format!("natural-n{shape}-d{:02}", index % 3 + 1);
        let relative_path = format!("tokens/{fixture_id}.i32le");
        let fixture_tokens = &document.tokens[..shape + CONTINUATION_TOKENS];
        let prompt = &fixture_tokens[..shape];
        let continuation = &fixture_tokens[shape..];
        let bytes = token_bytes(fixture_tokens);
        output_file(&fixture_dir.join(&relative_path), &bytes, check)?;
        fixture_tokens_for_exclusion.push((fixture_id.clone(), fixture_tokens.to_vec()));
        expected_paths.insert(relative_path.clone());
        let (_, prompt_sha256) = domain_token_sha256(&fixture_id, "prompt", prompt);
        let (_, continuation_sha256) =
            domain_token_sha256(&fixture_id, "continuation", continuation);
        natural.push(json!({
            "fixture_id": fixture_id,
            "document": document_record(document),
            "prompt_token_count": shape,
            "selected_suffix_tokens": shape - SCOPE_TOKENS,
            "continuation_token_count": CONTINUATION_TOKENS,
            "arm_order": ARM_ORDERS[index],
            "open_greedy_sentinel": index % 3 == 0,
            "reverse_replay": index == 11,
            "tokens": fixture_token_record(
                &fixture_id,
                "prompt_then_continuation",
                &relative_path,
                fixture_tokens,
            ),
            "prompt_sha256_domain_i32le": prompt_sha256,
            "continuation_sha256_domain_i32le": continuation_sha256,
            "terminal_feed_token_id": continuation[CONTINUATION_TOKENS - 1],
        }));
    }

    let scope_document = &tokenized[12];
    let scope_id = "scope-n2051";
    let scope_relative = format!("tokens/{scope_id}.i32le");
    let scope_tokens = &scope_document.tokens[..SCOPE_TOKENS];
    output_file(
        &fixture_dir.join(&scope_relative),
        &token_bytes(scope_tokens),
        check,
    )?;
    fixture_tokens_for_exclusion.push((scope_id.into(), scope_tokens.to_vec()));
    expected_paths.insert(scope_relative.clone());
    let scope = json!({
        "fixture_id": scope_id,
        "document": document_record(scope_document),
        "prompt_token_count": SCOPE_TOKENS,
        "selected_rows_expected": 0,
        "arm_order": "ABC",
        "tokens": fixture_token_record(scope_id, "prompt", &scope_relative, scope_tokens),
    });

    let retrieval_specs = [
        RetrievalSpec {
            fixture_id: "retrieval-single-0256",
            target: RetrievalKind::Single { position: 256 },
            decoy: RetrievalKind::Single { position: 3_584 },
            answer_tokens: 1,
        },
        RetrievalSpec {
            fixture_id: "retrieval-single-1280",
            target: RetrievalKind::Single { position: 1_280 },
            decoy: RetrievalKind::Single { position: 2_560 },
            answer_tokens: 2,
        },
        RetrievalSpec {
            fixture_id: "retrieval-single-2304",
            target: RetrievalKind::Single { position: 2_304 },
            decoy: RetrievalKind::Single { position: 512 },
            answer_tokens: 1,
        },
        RetrievalSpec {
            fixture_id: "retrieval-single-3584",
            target: RetrievalKind::Single { position: 3_584 },
            decoy: RetrievalKind::Single { position: 1_280 },
            answer_tokens: 2,
        },
        RetrievalSpec {
            fixture_id: "retrieval-double-0256-2304",
            target: RetrievalKind::Double {
                first: 256,
                second: 2_304,
            },
            decoy: RetrievalKind::Double {
                first: 1_024,
                second: 3_328,
            },
            answer_tokens: 1,
        },
        RetrievalSpec {
            fixture_id: "retrieval-double-0512-3584",
            target: RetrievalKind::Double {
                first: 512,
                second: 3_584,
            },
            decoy: RetrievalKind::Double {
                first: 1_536,
                second: 2_560,
            },
            answer_tokens: 2,
        },
        RetrievalSpec {
            fixture_id: "retrieval-double-1280-3072",
            target: RetrievalKind::Double {
                first: 1_280,
                second: 3_072,
            },
            decoy: RetrievalKind::Double {
                first: 256,
                second: 3_840,
            },
            answer_tokens: 1,
        },
        RetrievalSpec {
            fixture_id: "retrieval-double-1792-3584",
            target: RetrievalKind::Double {
                first: 1_792,
                second: 3_584,
            },
            decoy: RetrievalKind::Double {
                first: 512,
                second: 2_816,
            },
            answer_tokens: 2,
        },
    ];
    let mut used_answers = BTreeSet::new();
    let mut retrieval = Vec::new();
    let forbidden_answer_texts = tokenized
        .iter()
        .map(|document| document.text.as_str())
        .collect::<Vec<_>>();
    for (index, spec) in retrieval_specs.iter().enumerate() {
        let document = &tokenized[13 + index];
        let (answer, answer_tokens) = next_answer(
            &tokenizer,
            spec.answer_tokens,
            &mut used_answers,
            &forbidden_answer_texts,
        )?;
        let (decoy_answer, decoy_answer_tokens) = next_answer(
            &tokenizer,
            spec.answer_tokens,
            &mut used_answers,
            &forbidden_answer_texts,
        )?;
        let (prompt, mut record) = build_retrieval(
            &tokenizer,
            document,
            spec,
            &answer,
            &answer_tokens,
            &decoy_answer,
            &decoy_answer_tokens,
            &stop_tokens,
        )?;
        let relative_path = format!("tokens/{}.i32le", spec.fixture_id);
        output_file(
            &fixture_dir.join(&relative_path),
            &token_bytes(&prompt),
            check,
        )?;
        fixture_tokens_for_exclusion.push((spec.fixture_id.into(), prompt.clone()));
        expected_paths.insert(relative_path.clone());
        record.as_object_mut().unwrap().insert(
            "tokens".into(),
            fixture_token_record(spec.fixture_id, "prompt", &relative_path, &prompt),
        );
        record.as_object_mut().unwrap().insert(
            "arm_order".into(),
            Value::String(RETRIEVAL_ARM_ORDERS[index].into()),
        );
        retrieval.push(record);
    }

    let prior_token_fixtures =
        source_plan.selection["repository_exclusion_audit"]["token_fixture_inventory"]
            .as_array()
            .context("repository token fixture inventory")?;
    let mut prior_grams = Vec::new();
    for prior in prior_token_fixtures {
        let path = prior["path"].as_str().context("prior token fixture path")?;
        let token_ids = prior["token_ids_u32"]
            .as_array()
            .context("prior token fixture IDs")?
            .iter()
            .map(|value| {
                let token = value.as_u64().context("prior token ID")?;
                i32::try_from(token).context("prior token ID exceeds i32")
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = token_bytes(&token_ids);
        ensure!(bytes.len() == prior["bytes"].as_u64().unwrap() as usize);
        ensure!(sha256(&bytes) == prior["sha256"].as_str().unwrap());
        prior_grams.push((path.to_string(), token_five_grams(&token_ids)));
    }
    let mut maximum_common_five_grams = 0_usize;
    let mut maximum_pair = None;
    for (fixture_id, tokens) in &fixture_tokens_for_exclusion {
        let fixture_grams = token_five_grams(tokens);
        for (prior_path, grams) in &prior_grams {
            let common = fixture_grams.intersection(grams).count();
            if common > maximum_common_five_grams {
                maximum_common_five_grams = common;
                maximum_pair = Some(json!([fixture_id, prior_path]));
            }
        }
    }
    ensure!(
        maximum_common_five_grams == 0,
        "frozen fixture overlaps a prior repository token prompt"
    );
    source_plan.selection.as_object_mut().unwrap().insert(
        "repository_token_fixture_exclusion_audit".into(),
        json!({
            "algorithm": "exact_token_id_5gram_set_intersection_v1",
            "current_fixture_count": fixture_tokens_for_exclusion.len(),
            "prior_fixture_count": prior_grams.len(),
            "comparison_count": fixture_tokens_for_exclusion.len() * prior_grams.len(),
            "maximum_common_five_grams": maximum_common_five_grams,
            "maximum_pair": maximum_pair,
            "required_maximum_common_five_grams": 0,
            "passed": true,
        }),
    );

    let mut operation_plan = Vec::new();
    push_arm_operations(
        &mut operation_plan,
        "scope_control",
        scope_id,
        "prefill_endpoint_replay",
        "ABC",
    )?;
    for fixture in &natural {
        push_arm_operations(
            &mut operation_plan,
            "natural_semantic",
            fixture["fixture_id"].as_str().unwrap(),
            "teacher_forced_nll_96",
            fixture["arm_order"].as_str().unwrap(),
        )?;
    }
    for fixture in natural
        .iter()
        .filter(|fixture| fixture["open_greedy_sentinel"] == true)
    {
        push_arm_operations(
            &mut operation_plan,
            "open_greedy",
            fixture["fixture_id"].as_str().unwrap(),
            "greedy_32_or_stop",
            fixture["arm_order"].as_str().unwrap(),
        )?;
    }
    for fixture in &retrieval {
        push_arm_operations(
            &mut operation_plan,
            "retrieval_semantic",
            fixture["fixture_id"].as_str().unwrap(),
            "answer_nll_and_exact_prefix",
            fixture["arm_order"].as_str().unwrap(),
        )?;
    }
    let reverse_fixture = natural
        .iter()
        .find(|fixture| fixture["reverse_replay"] == true)
        .context("missing reverse replay fixture")?;
    let reverse_order = reverse_fixture["arm_order"]
        .as_str()
        .unwrap()
        .chars()
        .rev()
        .collect::<String>();
    push_arm_operations(
        &mut operation_plan,
        "reverse_replay",
        reverse_fixture["fixture_id"].as_str().unwrap(),
        "teacher_forced_nll_96_replay",
        &reverse_order,
    )?;

    let mut llama_operations = natural
        .iter()
        .map(|fixture| {
            (
                fixture["fixture_id"].as_str().unwrap(),
                "teacher_forced_nll_96",
                fixture["tokens"]["sha256_raw_i32le"].as_str().unwrap(),
            )
        })
        .chain(retrieval.iter().map(|fixture| {
            (
                fixture["fixture_id"].as_str().unwrap(),
                "answer_nll_and_exact_prefix",
                fixture["tokens"]["sha256_raw_i32le"].as_str().unwrap(),
            )
        }))
        .map(|(fixture_id, mode, token_sha256)| {
            let ordering_key = sha256(
                format!(
                    "qwen4exp-selected-quality-llama-order-v1\0fixture_id={fixture_id}\ntokens_sha256={token_sha256}\n"
                )
                .as_bytes(),
            );
            (ordering_key, fixture_id, mode)
        })
        .collect::<Vec<_>>();
    llama_operations.sort();
    for (ordering_key, fixture_id, mode) in llama_operations {
        operation_plan.push(json!({
            "ordinal": operation_plan.len(),
            "phase": "llama_cpp_triangulation",
            "fixture_id": fixture_id,
            "mode": mode,
            "arm": "D",
            "ordering_key_sha256": ordering_key,
        }));
    }

    if tokens_dir.exists() {
        for entry in std::fs::read_dir(&tokens_dir).context("read token fixture directory")? {
            let entry = entry?;
            let relative = format!("tokens/{}", entry.file_name().to_string_lossy());
            ensure!(
                expected_paths.contains(&relative),
                "unexpected generated token fixture {relative}"
            );
        }
    }

    let preregistration_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/README.md");
    let preregistration_bytes = std::fs::read(&preregistration_path)
        .with_context(|| format!("read preregistration {}", preregistration_path.display()))?;
    let generator_bytes = include_bytes!("qwen4exp_selected_quality_prepare.rs");
    let mut model_manifest_domain = String::from("qwen4exp-release-model-shard-manifest-v1\0");
    let model_shards = MODEL_SHARDS
        .iter()
        .enumerate()
        .map(|(index, &(filename, bytes, sha256))| {
            model_manifest_domain.push_str(&format!("{index}\t{filename}\t{bytes}\t{sha256}\n"));
            json!({
                "index": index,
                "filename": filename,
                "bytes": bytes,
                "sha256": sha256,
            })
        })
        .collect::<Vec<_>>();
    let model_manifest_sha256 = sha256(model_manifest_domain.as_bytes());
    let manifest = json!({
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "packet_id": "2026-08-28-qwen4exp-selected-quality-v2",
        "status": "fixtures_frozen_not_acquired",
        "preregistration": {
            "path": "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/README.md",
            "bytes": preregistration_bytes.len(),
            "sha256": sha256(&preregistration_bytes),
        },
        "generator": {
            "path": "crates/qwen-llm/examples/qwen4exp_selected_quality_prepare.rs",
            "bytes": generator_bytes.len(),
            "sha256": sha256(generator_bytes),
            "source_plan_path": source_plan_path,
            "source_plan_bytes": source_plan_bytes.len(),
            "source_plan_sha256": sha256(&source_plan_bytes),
        },
        "acquisition_model_lock": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "quant": "UD-Q3_K_XL",
            "fixture_generation_dependency": "none beyond the separately qualified tokenizer artifact",
            "verification_phase": "hash every local shard before local or llama.cpp acquisition",
            "shard_manifest_schema": "qwen4exp-release-model-shard-manifest-v1",
            "shard_manifest_domain_utf8": model_manifest_domain,
            "shard_manifest_sha256": model_manifest_sha256,
            "shards": model_shards,
        },
        "tokenizer": {
            "identity_schema": "qwen4exp-tokenizer-identity-v2",
            "identity_sha256": hex(&tokenizer_identity),
            "vocab_size": tokenizer.n_vocab(),
            "producer_stop_token_ids": stop_tokens,
            "model": gguf.get_str("tokenizer.ggml.model"),
            "pretokenizer": gguf.get_str("tokenizer.ggml.pre"),
            "chat_template_sha256": gguf
                .get_str("tokenizer.chat_template")
                .map(|template| sha256(template.as_bytes())),
            "add_special_tokens": false,
            "fixture_generation_qualification": "exact released tokenizer fingerprint; model weight bytes are not fixture inputs",
        },
        "corpus": {
            "source": source_plan.source,
            "selection": source_plan.selection,
            "token_qualification": {
                "minimum_tokens": MAX_REQUIRED_DOCUMENT_TOKENS,
                "qualified_documents": token_qualified_documents,
                "selected_documents": 21,
                "selection_order": "first token-qualified documents in source row order",
            },
        },
        "execution": {
            "local_arms": ["A_default_safe", "B_generic_selected", "C_f32_hc_down"],
            "orders": ARM_ORDERS,
            "each_permutation_count": 2,
            "operation_plan": operation_plan,
            "support_runs": {
                "included": false,
                "reason": "selector captures are localization evidence acquired only in a separately preregistered packet",
            },
            "llama_cpp": {
                "arm": "D_same_manifest_tokens_separate_order",
                "repository": "ggml-org/llama.cpp",
                "commit": LLAMA_CPP_COMMIT,
                "support_pull_request": 27742,
            },
            "bootstrap": {
                "unit": "document",
                "strata": [2179, 2563, 3075, 4099],
                "documents_per_stratum": 3,
                "draws": 100000,
                "rng": BOOTSTRAP_RNG,
                "seed_u64": BOOTSTRAP_SEED,
                "draw_order": "draw-major, then ascending stratum, then three replacement indices; one shared resample matrix for B-A, C-A, and C-B",
                "bounded_mapping": "largest u64 acceptance zone divisible by 3, reject values outside it, accepted_value modulo 3",
                "aggregation": "mean of the 12 sampled document mean-NLL deltas; each document contains 96 equally weighted tokens",
                "quantiles": {
                    "candidate_vs_incumbent": 0.975,
                    "challenger_vs_generic": 0.95,
                },
                "quantile_convention": "nearest-rank upper percentile: sorted_draws[ceil(p * draws) - 1], no interpolation",
            },
        },
        "natural_fixtures": natural,
        "scope_control": scope,
        "retrieval_fixtures": retrieval,
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let mut manifest_with_newline = manifest_bytes;
    manifest_with_newline.push(b'\n');
    output_file(
        &fixture_dir.join("fixtures.json"),
        &manifest_with_newline,
        check,
    )?;

    println!(
        "fixture_dir={} manifest_bytes={} manifest_sha256={} mode={}",
        fixture_dir.display(),
        manifest_with_newline.len(),
        sha256(&manifest_with_newline),
        if check { "check" } else { "write" },
    );
    Ok(())
}
