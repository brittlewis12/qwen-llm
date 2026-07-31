//! CPU-only oracle for token-level grammar topology.
//!
//! The oracle analyzes exact decoded token bytes. It does not load model
//! weights, execute Metal work, or estimate model probabilities.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_llm::gguf::GgufFile;
use qwen_llm::tokenizer::NativeTokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const OUTPUT_SCHEMA: &str = "grammar-trace-oracle/v1";
const MAX_LANGUAGE_STRINGS: usize = 4096;
const MAX_LANGUAGE_STRING_BYTES: usize = 64 * 1024;
const FAST_FORWARD_MIN_RUN: usize = 4;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-grammar-oracle",
    version,
    about = "analyze exact token-level grammar topology without model execution"
)]
struct Args {
    /// Qwen GGUF supplying the tokenizer metadata.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Ordered-enum finite-language grammar specification.
    #[arg(long)]
    grammar: PathBuf,
    /// Frozen real-output trace fixture used only for path weighting.
    #[arg(long)]
    traces: PathBuf,
    /// Write pretty JSON here instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Measured dense-27B serial transition cost.
    #[arg(long)]
    serial_transition_ms: f64,
    /// Measured dense-27B physical-N8 verifier packet cost.
    #[arg(long)]
    fixed_n8_packet_ms: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GrammarSpec {
    schema_version: u32,
    grammar_id: String,
    kind: String,
    fields: Vec<EnumField>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EnumField {
    name: String,
    values: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TraceFixture {
    schema_version: u32,
    fixture_id: String,
    source: TraceSource,
    records: Vec<TraceRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TraceSource {
    source_model: String,
    results_path: String,
    results_sha256: String,
    prompt_builder_path: String,
    prompt_builder_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TraceRecord {
    item_id: String,
    branch: String,
    gen_text: String,
    gen_tokens: Vec<i32>,
    n_prompt: usize,
    rendered_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct FieldChoice {
    field: String,
    value: String,
}

#[derive(Clone, Debug)]
struct LanguageMember {
    text: String,
    bytes: Vec<u8>,
    choices: Vec<FieldChoice>,
}

#[derive(Clone, Debug)]
struct TokenPiece {
    id: i32,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct TokenIndex {
    by_id: Vec<Option<Vec<u8>>>,
    by_bytes: BTreeMap<Vec<u8>, Vec<i32>>,
    pieces: Vec<TokenPiece>,
}

#[derive(Debug)]
struct TokenInventory {
    index: TokenIndex,
    grammar_piece_policy_sha256: String,
    admissible_nonempty_rows: usize,
    ordinary_content_empty_rows: usize,
    excluded_noncontent_nonempty_rows: usize,
    excluded_noncontent_empty_rows: usize,
    duplicate_piece_groups: usize,
    max_ids_per_piece: usize,
}

#[derive(Clone, Debug)]
struct StateAnalysis {
    prefix: Vec<u8>,
    terminal: bool,
    admissible: Vec<i32>,
    reachable: bool,
    forced_suffix_len: usize,
}

#[derive(Debug)]
struct TopologyAnalysis {
    states: Vec<StateAnalysis>,
    by_prefix: BTreeMap<Vec<u8>, usize>,
    indexed_naive_cross_checks: usize,
    state_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalPath {
    text: String,
    text_sha256: String,
    choices: Vec<FieldChoice>,
    token_ids: Vec<i32>,
    admissible_rows_before_token: Vec<usize>,
    forced_runs: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct OutputDocument {
    schema: &'static str,
    decision: Value,
    claim_boundary: Value,
    inputs: Value,
    tokenizer: Value,
    grammar: Value,
    global_topology: Value,
    canonical_paths: Value,
    real_trace_weighting: Value,
    counterfactual_fixed_n8_screen: Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CursorCase {
    AlignedNonterminal,
    AlignedDirectTerminal,
    LeadingPendingTerminal,
    LeadingPendingNonterminal,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bytes_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<(T, Vec<u8>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
    let value = serde_json::from_slice(&bytes).with_context(|| format!("parse {path:?}"))?;
    Ok((value, bytes))
}

fn validate_grammar(spec: &GrammarSpec) -> Result<()> {
    ensure!(spec.schema_version == 1, "grammar schema_version must be 1");
    ensure!(
        spec.kind == "ordered_enum_object",
        "unsupported grammar kind {:?}",
        spec.kind
    );
    ensure!(!spec.grammar_id.is_empty(), "grammar_id must not be empty");
    ensure!(!spec.fields.is_empty(), "grammar must contain fields");

    let mut field_names = BTreeSet::new();
    let mut combinations = 1usize;
    for field in &spec.fields {
        ensure!(
            !field.name.is_empty(),
            "grammar field name must not be empty"
        );
        ensure!(
            field_names.insert(field.name.as_str()),
            "duplicate grammar field {:?}",
            field.name
        );
        ensure!(
            !field.values.is_empty(),
            "grammar field {:?} has no values",
            field.name
        );
        let unique: BTreeSet<_> = field.values.iter().collect();
        ensure!(
            unique.len() == field.values.len(),
            "grammar field {:?} has duplicate values",
            field.name
        );
        combinations = combinations
            .checked_mul(field.values.len())
            .context("grammar language cardinality overflow")?;
        ensure!(
            combinations <= MAX_LANGUAGE_STRINGS,
            "grammar expands to more than {MAX_LANGUAGE_STRINGS} strings"
        );
    }
    Ok(())
}

fn render_member(spec: &GrammarSpec, selected: &[usize]) -> Result<LanguageMember> {
    ensure!(
        selected.len() == spec.fields.len(),
        "selection arity mismatch"
    );
    let mut text = String::from("{");
    let mut choices = Vec::with_capacity(spec.fields.len());
    for (index, (field, &value_index)) in spec.fields.iter().zip(selected).enumerate() {
        let value = field
            .values
            .get(value_index)
            .context("selection value index out of range")?;
        if index != 0 {
            text.push(',');
        }
        text.push_str(&serde_json::to_string(&field.name)?);
        text.push(':');
        text.push_str(&serde_json::to_string(value)?);
        choices.push(FieldChoice {
            field: field.name.clone(),
            value: value.clone(),
        });
    }
    text.push('}');
    ensure!(
        text.len() <= MAX_LANGUAGE_STRING_BYTES,
        "rendered grammar member exceeds {MAX_LANGUAGE_STRING_BYTES} bytes"
    );
    Ok(LanguageMember {
        bytes: text.as_bytes().to_vec(),
        text,
        choices,
    })
}

fn generate_language(spec: &GrammarSpec) -> Result<Vec<LanguageMember>> {
    validate_grammar(spec)?;
    let mut selected = vec![0usize; spec.fields.len()];
    let mut members = Vec::new();

    fn visit(
        spec: &GrammarSpec,
        depth: usize,
        selected: &mut [usize],
        members: &mut Vec<LanguageMember>,
    ) -> Result<()> {
        if depth == spec.fields.len() {
            members.push(render_member(spec, selected)?);
            return Ok(());
        }
        for value_index in 0..spec.fields[depth].values.len() {
            selected[depth] = value_index;
            visit(spec, depth + 1, selected, members)?;
        }
        Ok(())
    }

    visit(spec, 0, &mut selected, &mut members)?;
    members.sort_by(|left, right| left.bytes.cmp(&right.bytes));
    for pair in members.windows(2) {
        ensure!(pair[0].bytes != pair[1].bytes, "duplicate language member");
        ensure!(
            !pair[1].bytes.starts_with(&pair[0].bytes),
            "language must be prefix-free: {:?} prefixes {:?}",
            pair[0].text,
            pair[1].text
        );
    }
    Ok(members)
}

impl TokenIndex {
    fn new(vocab_size: usize, mut pieces: Vec<TokenPiece>) -> Result<Self> {
        pieces.sort_by_key(|piece| piece.id);
        let mut by_id = vec![None; vocab_size];
        let mut by_bytes: BTreeMap<Vec<u8>, Vec<i32>> = BTreeMap::new();
        for piece in &pieces {
            ensure!(!piece.bytes.is_empty(), "empty pieces must be excluded");
            let index = usize::try_from(piece.id).context("negative token id")?;
            ensure!(index < vocab_size, "token id {} is out of range", piece.id);
            ensure!(by_id[index].is_none(), "duplicate token id {}", piece.id);
            by_id[index] = Some(piece.bytes.clone());
            by_bytes
                .entry(piece.bytes.clone())
                .or_default()
                .push(piece.id);
        }
        for ids in by_bytes.values_mut() {
            ids.sort_unstable();
        }
        Ok(Self {
            by_id,
            by_bytes,
            pieces,
        })
    }

    fn piece(&self, token: i32) -> Option<&[u8]> {
        let index = usize::try_from(token).ok()?;
        self.by_id.get(index)?.as_deref()
    }
}

fn build_token_inventory(tokenizer: &NativeTokenizer) -> Result<TokenInventory> {
    let vocab_size = tokenizer.n_vocab() as usize;
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-token-piece-table/v1\0");
    let mut pieces = Vec::new();
    let mut ordinary_content_empty_rows = 0usize;
    let mut excluded_noncontent_nonempty_rows = 0usize;
    let mut excluded_noncontent_empty_rows = 0usize;

    for token in 0..vocab_size {
        let token = i32::try_from(token).context("vocab id exceeds i32")?;
        let bytes = tokenizer.try_decode_piece_bytes_exact(token)?;
        let ordinary = tokenizer.is_ordinary_content_token(token)?;
        hasher.update(token.to_le_bytes());
        hasher.update([u8::from(ordinary)]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
        if ordinary {
            if bytes.is_empty() {
                ordinary_content_empty_rows += 1;
            } else {
                pieces.push(TokenPiece {
                    id: token,
                    bytes: bytes.to_vec(),
                });
            }
        } else if bytes.is_empty() {
            excluded_noncontent_empty_rows += 1;
        } else {
            excluded_noncontent_nonempty_rows += 1;
        }
    }

    ensure!(
        pieces.len()
            + ordinary_content_empty_rows
            + excluded_noncontent_nonempty_rows
            + excluded_noncontent_empty_rows
            == vocab_size,
        "token inventory partition does not cover the vocabulary"
    );

    let index = TokenIndex::new(vocab_size, pieces)?;
    let duplicate_piece_groups = index.by_bytes.values().filter(|ids| ids.len() > 1).count();
    let max_ids_per_piece = index.by_bytes.values().map(Vec::len).max().unwrap_or(0);
    Ok(TokenInventory {
        admissible_nonempty_rows: index.pieces.len(),
        index,
        grammar_piece_policy_sha256: format!("{:x}", hasher.finalize()),
        ordinary_content_empty_rows,
        excluded_noncontent_nonempty_rows,
        excluded_noncontent_empty_rows,
        duplicate_piece_groups,
        max_ids_per_piece,
    })
}

fn productive_prefixes(members: &[LanguageMember]) -> BTreeSet<Vec<u8>> {
    let mut prefixes = BTreeSet::new();
    for member in members {
        for end in 0..=member.bytes.len() {
            prefixes.insert(member.bytes[..end].to_vec());
        }
    }
    prefixes
}

fn indexed_admissible(prefix: &[u8], members: &[LanguageMember], index: &TokenIndex) -> Vec<i32> {
    let mut ids = BTreeSet::new();
    for member in members {
        if !member.bytes.starts_with(prefix) {
            continue;
        }
        let remainder = &member.bytes[prefix.len()..];
        for end in 1..=remainder.len() {
            if let Some(tokens) = index.by_bytes.get(&remainder[..end]) {
                ids.extend(tokens.iter().copied());
            }
        }
    }
    ids.into_iter().collect()
}

fn naive_admissible(prefix: &[u8], members: &[LanguageMember], index: &TokenIndex) -> Vec<i32> {
    let suffixes: Vec<&[u8]> = members
        .iter()
        .filter(|member| member.bytes.starts_with(prefix))
        .map(|member| &member.bytes[prefix.len()..])
        .collect();
    index
        .pieces
        .iter()
        .filter(|piece| {
            suffixes
                .iter()
                .any(|suffix| suffix.starts_with(&piece.bytes))
        })
        .map(|piece| piece.id)
        .collect()
}

fn state_digest(states: &[StateAnalysis]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-state-analysis/v1\0");
    for state in states {
        hasher.update((state.prefix.len() as u64).to_le_bytes());
        hasher.update(&state.prefix);
        hasher.update([u8::from(state.terminal), u8::from(state.reachable)]);
        hasher.update((state.forced_suffix_len as u64).to_le_bytes());
        hasher.update((state.admissible.len() as u64).to_le_bytes());
        for token in &state.admissible {
            hasher.update(token.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn analyze_topology(members: &[LanguageMember], index: &TokenIndex) -> Result<TopologyAnalysis> {
    let prefixes = productive_prefixes(members);
    let terminals: BTreeSet<Vec<u8>> = members.iter().map(|member| member.bytes.clone()).collect();
    let mut states = Vec::with_capacity(prefixes.len());
    let mut by_prefix = BTreeMap::new();
    let mut indexed_naive_cross_checks = 0usize;

    for prefix in prefixes {
        let indexed = indexed_admissible(&prefix, members, index);
        let naive = naive_admissible(&prefix, members, index);
        ensure!(
            indexed == naive,
            "indexed/naive admissibility mismatch at prefix {}",
            bytes_hex(&prefix)
        );
        indexed_naive_cross_checks += 1;
        let state_index = states.len();
        by_prefix.insert(prefix.clone(), state_index);
        states.push(StateAnalysis {
            terminal: terminals.contains(&prefix),
            prefix,
            admissible: indexed,
            reachable: false,
            forced_suffix_len: 0,
        });
    }

    let root = *by_prefix.get(&Vec::new()).context("language has no root")?;
    states[root].reachable = true;
    let mut queue = VecDeque::from([root]);
    while let Some(state_index) = queue.pop_front() {
        let prefix = states[state_index].prefix.clone();
        let admissible = states[state_index].admissible.clone();
        for token in admissible {
            let piece = index
                .piece(token)
                .context("admissible token is absent from token index")?;
            let mut next = prefix.clone();
            next.extend_from_slice(piece);
            let &next_index = by_prefix
                .get(&next)
                .context("admissible edge does not reach productive prefix")?;
            if !states[next_index].reachable {
                states[next_index].reachable = true;
                queue.push_back(next_index);
            }
        }
    }

    let mut order: Vec<usize> = (0..states.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(states[index].prefix.len()));
    for state_index in order {
        if states[state_index].terminal || states[state_index].admissible.len() != 1 {
            continue;
        }
        let token = states[state_index].admissible[0];
        let piece = index
            .piece(token)
            .context("singleton token is absent from token index")?;
        let mut next = states[state_index].prefix.clone();
        next.extend_from_slice(piece);
        let &next_index = by_prefix
            .get(&next)
            .context("singleton edge does not reach productive prefix")?;
        states[state_index].forced_suffix_len = 1 + states[next_index].forced_suffix_len;
    }

    let state_sha256 = state_digest(&states);
    Ok(TopologyAnalysis {
        states,
        by_prefix,
        indexed_naive_cross_checks,
        state_sha256,
    })
}

fn push_run(runs: &mut Vec<usize>, current: &mut usize) {
    if *current != 0 {
        runs.push(*current);
        *current = 0;
    }
}

fn analyze_token_path(
    member: &LanguageMember,
    token_ids: Vec<i32>,
    index: &TokenIndex,
    topology: &TopologyAnalysis,
) -> Result<CanonicalPath> {
    ensure!(!token_ids.is_empty(), "canonical path is empty");
    let mut prefix = Vec::new();
    let mut decoded = Vec::new();
    let mut row_counts = Vec::with_capacity(token_ids.len());
    let mut forced_runs = Vec::new();
    let mut current_run = 0usize;
    for &token in &token_ids {
        let piece = index
            .piece(token)
            .with_context(|| format!("canonical token {token} is not ordinary nonempty content"))?;
        let &state_index = topology
            .by_prefix
            .get(&prefix)
            .context("canonical path left grammar state graph")?;
        let state = &topology.states[state_index];
        ensure!(
            state.admissible.binary_search(&token).is_ok(),
            "canonical token {token} is inadmissible at prefix {}",
            bytes_hex(&prefix)
        );
        row_counts.push(state.admissible.len());
        if state.admissible.len() == 1 {
            current_run += 1;
        } else {
            push_run(&mut forced_runs, &mut current_run);
        }
        prefix.extend_from_slice(piece);
        decoded.extend_from_slice(piece);
    }
    push_run(&mut forced_runs, &mut current_run);
    ensure!(
        decoded == member.bytes,
        "canonical raw-byte round trip failed"
    );
    let &terminal_index = topology
        .by_prefix
        .get(&prefix)
        .context("canonical path has no terminal state")?;
    ensure!(
        topology.states[terminal_index].terminal,
        "canonical path did not end at grammar terminal"
    );
    Ok(CanonicalPath {
        text: member.text.clone(),
        text_sha256: sha256_hex(&member.bytes),
        choices: member.choices.clone(),
        token_ids,
        admissible_rows_before_token: row_counts,
        forced_runs,
    })
}

fn analyze_canonical_paths(
    members: &[LanguageMember],
    tokenizer: &NativeTokenizer,
    index: &TokenIndex,
    topology: &TopologyAnalysis,
) -> Result<Vec<CanonicalPath>> {
    let mut paths = Vec::with_capacity(members.len());
    for member in members {
        let token_ids = tokenizer
            .encode(&member.text, false)
            .with_context(|| format!("canonical-tokenize {:?}", member.text))?;
        paths.push(analyze_token_path(member, token_ids, index, topology)?);
    }
    Ok(paths)
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut out = BTreeMap::new();
    for value in values {
        *out.entry(value).or_insert(0) += 1;
    }
    out
}

fn nearest_rank(values: &[usize], numerator: usize, denominator: usize) -> Option<usize> {
    if values.is_empty() || denominator == 0 || numerator > denominator {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = numerator
        .saturating_mul(sorted.len())
        .div_ceil(denominator)
        .max(1);
    sorted.get(rank - 1).copied()
}

fn quantiles(values: &[usize]) -> Value {
    json!({
        "convention": "nearest_rank",
        "count": values.len(),
        "min": values.iter().min(),
        "p50": nearest_rank(values, 50, 100),
        "p90": nearest_rank(values, 90, 100),
        "p95": nearest_rank(values, 95, 100),
        "max": values.iter().max(),
    })
}

fn canonical_path_digest(paths: &[CanonicalPath]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-canonical-paths/v1\0");
    for path in paths {
        hasher.update((path.text.len() as u64).to_le_bytes());
        hasher.update(path.text.as_bytes());
        hasher.update((path.token_ids.len() as u64).to_le_bytes());
        for token in &path.token_ids {
            hasher.update(token.to_le_bytes());
        }
        for rows in &path.admissible_rows_before_token {
            hasher.update((*rows as u64).to_le_bytes());
        }
        for run in &path.forced_runs {
            hasher.update((*run as u64).to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn state_subset_digest(states: &[&StateAnalysis]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-state-subset/v1\0");
    for state in states {
        hasher.update((state.prefix.len() as u64).to_le_bytes());
        hasher.update(&state.prefix);
        hasher.update((state.forced_suffix_len as u64).to_le_bytes());
        hasher.update((state.admissible.len() as u64).to_le_bytes());
        for token in &state.admissible {
            hasher.update(token.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn token_id_set_digest(tokens: &BTreeSet<i32>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-token-id-set/v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn required_transitions(run_tokens: usize, case: CursorCase) -> usize {
    match case {
        CursorCase::AlignedNonterminal => run_tokens,
        CursorCase::AlignedDirectTerminal => run_tokens.saturating_sub(1),
        CursorCase::LeadingPendingTerminal => run_tokens,
        CursorCase::LeadingPendingNonterminal => run_tokens.saturating_add(1),
    }
}

fn rounded_ms(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn summarize_trace_rows(
    records: &[&TraceRecord],
    paths_by_text: &BTreeMap<&str, &CanonicalPath>,
) -> Result<Value> {
    let mut rows = Vec::new();
    let mut runs = Vec::new();
    let mut outputs = BTreeSet::new();
    let mut original_counts = Vec::new();
    let mut final_ids = BTreeMap::new();
    for record in records {
        let path = paths_by_text
            .get(record.gen_text.as_str())
            .context("trace output is outside grammar language")?;
        rows.extend_from_slice(&path.admissible_rows_before_token);
        runs.extend_from_slice(&path.forced_runs);
        outputs.insert(record.gen_text.as_str());
        original_counts.push(record.gen_tokens.len());
        if let Some(&token) = record.gen_tokens.last() {
            *final_ids.entry(token).or_insert(0usize) += 1;
        }
    }
    Ok(json!({
        "records": records.len(),
        "distinct_outputs": outputs.len(),
        "canonical_retokenized_admissible_rows": quantiles(&rows),
        "canonical_retokenized_forced_run_histogram": histogram(runs),
        "original_source_token_count_histogram": histogram(original_counts),
        "original_source_final_token_id_histogram": final_ids,
    }))
}

fn validate_trace_fixture(fixture: &TraceFixture, members: &[LanguageMember]) -> Result<()> {
    ensure!(
        fixture.schema_version == 1,
        "trace schema_version must be 1"
    );
    ensure!(
        !fixture.fixture_id.is_empty(),
        "trace fixture_id must not be empty"
    );
    ensure!(!fixture.records.is_empty(), "trace fixture has no records");
    ensure!(
        valid_sha256(&fixture.source.results_sha256),
        "invalid source results_sha256"
    );
    ensure!(
        valid_sha256(&fixture.source.prompt_builder_sha256),
        "invalid source prompt_builder_sha256"
    );
    let valid_texts: BTreeSet<&str> = members.iter().map(|member| member.text.as_str()).collect();
    let mut keys = BTreeSet::new();
    for record in &fixture.records {
        ensure!(
            !record.item_id.is_empty(),
            "trace item_id must not be empty"
        );
        ensure!(!record.branch.is_empty(), "trace branch must not be empty");
        ensure!(
            valid_texts.contains(record.gen_text.as_str()),
            "trace output is outside grammar language: {:?}",
            record.gen_text
        );
        ensure!(
            !record.gen_tokens.is_empty(),
            "trace source token list is empty"
        );
        ensure!(
            valid_sha256(&record.rendered_sha256),
            "invalid rendered_sha256 for {:?}",
            record.item_id
        );
        ensure!(
            keys.insert((record.item_id.as_str(), record.branch.as_str())),
            "duplicate trace item/branch pair"
        );
    }
    Ok(())
}

fn prefix_json(prefix: &[u8]) -> Value {
    json!({
        "byte_length": prefix.len(),
        "hex": bytes_hex(prefix),
        "utf8": std::str::from_utf8(prefix).ok(),
    })
}

fn language_digest(members: &[LanguageMember]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-finite-language/v1\0");
    for member in members {
        hasher.update((member.bytes.len() as u64).to_le_bytes());
        hasher.update(&member.bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn build_output(
    args: &Args,
    grammar: &GrammarSpec,
    grammar_bytes: &[u8],
    traces: &TraceFixture,
    trace_bytes: &[u8],
    stop_token_ids: &[i32],
    inventory: &TokenInventory,
    members: &[LanguageMember],
    topology: &TopologyAnalysis,
    canonical_paths: &[CanonicalPath],
) -> Result<OutputDocument> {
    let reachable: Vec<_> = topology
        .states
        .iter()
        .filter(|state| state.reachable)
        .collect();
    let nonterminal: Vec<_> = reachable
        .iter()
        .copied()
        .filter(|state| !state.terminal)
        .collect();
    let global_rows: Vec<usize> = nonterminal
        .iter()
        .map(|state| state.admissible.len())
        .collect();
    let branch_rows: Vec<usize> = nonterminal
        .iter()
        .map(|state| state.admissible.len())
        .filter(|&rows| rows > 1)
        .collect();
    let unique_admissible_tokens: BTreeSet<i32> = nonterminal
        .iter()
        .flat_map(|state| state.admissible.iter().copied())
        .collect();
    let unique_branch_tokens: BTreeSet<i32> = nonterminal
        .iter()
        .filter(|state| state.admissible.len() > 1)
        .flat_map(|state| state.admissible.iter().copied())
        .collect();
    let state_token_incidences: usize = global_rows.iter().sum();
    let branch_state_token_incidences: usize = branch_rows.iter().sum();
    let max_forced_run = reachable
        .iter()
        .map(|state| state.forced_suffix_len)
        .max()
        .unwrap_or(0);
    let states_attaining_maximum: Vec<&StateAnalysis> = reachable
        .iter()
        .copied()
        .filter(|state| state.forced_suffix_len == max_forced_run && max_forced_run != 0)
        .collect();
    let maximum_state_examples: Vec<Value> = states_attaining_maximum
        .iter()
        .take(8)
        .map(|state| {
            let token = state.admissible[0];
            json!({
                "prefix": prefix_json(&state.prefix),
                "token_id": token,
                "piece": prefix_json(
                    inventory.index.piece(token).expect("indexed singleton token")
                ),
            })
        })
        .collect();

    let canonical_rows: Vec<usize> = canonical_paths
        .iter()
        .flat_map(|path| path.admissible_rows_before_token.iter().copied())
        .collect();
    let canonical_runs: Vec<usize> = canonical_paths
        .iter()
        .flat_map(|path| path.forced_runs.iter().copied())
        .collect();
    let canonical_token_counts = canonical_paths.iter().map(|path| path.token_ids.len());
    let canonical_forced_run_count = canonical_runs.len();
    let canonical_maximum_forced_run = canonical_runs.iter().max().copied().unwrap_or(0);
    let canonical_forced_token_positions: usize = canonical_runs.iter().sum();

    let paths_by_text: BTreeMap<&str, &CanonicalPath> = canonical_paths
        .iter()
        .map(|path| (path.text.as_str(), path))
        .collect();
    let mut by_branch: BTreeMap<&str, Vec<&TraceRecord>> = BTreeMap::new();
    for record in &traces.records {
        by_branch.entry(&record.branch).or_default().push(record);
    }
    let mut trace_strata = serde_json::Map::new();
    for (branch, records) in by_branch {
        trace_strata.insert(
            branch.to_string(),
            summarize_trace_rows(&records, &paths_by_text)?,
        );
    }
    let all_records: Vec<&TraceRecord> = traces.records.iter().collect();

    let max_rows = global_rows.iter().max().copied().unwrap_or(0);
    let vocab_rows = inventory.index.by_id.len();
    let max_row_fraction = max_rows as f64 / vocab_rows as f64;
    let row_screen_limit = (inventory.index.by_id.len() * 20) / 100;
    let fast_forward_disposition = if max_forced_run < FAST_FORWARD_MIN_RUN {
        "KILL_MAX_RUN_BELOW_LOCAL_FLOOR"
    } else {
        "BLOCKED_MISSING_CPACK_AND_SCAN_COST"
    };
    let row_disposition = if max_rows <= row_screen_limit {
        "POSITIVE_TOPOLOGY_SIGNAL_ONLY"
    } else {
        "KILL_ROW_PRUNING_FLOOR"
    };

    let cursor_cases = [
        ("aligned_nonterminal", CursorCase::AlignedNonterminal),
        ("aligned_direct_terminal", CursorCase::AlignedDirectTerminal),
        (
            "leading_pending_terminal",
            CursorCase::LeadingPendingTerminal,
        ),
        (
            "leading_pending_nonterminal",
            CursorCase::LeadingPendingNonterminal,
        ),
    ];
    let cursor_rows: Vec<Value> = cursor_cases
        .iter()
        .map(|(label, case)| {
            let transitions = required_transitions(max_forced_run, *case);
            json!({
                "case": label,
                "required_transitions": transitions,
                "fixed_n8_saving_ms": rounded_ms(
                    transitions as f64 * args.serial_transition_ms
                        - args.fixed_n8_packet_ms
                ),
            })
        })
        .collect();

    Ok(OutputDocument {
        schema: OUTPUT_SCHEMA,
        decision: json!({
            "forced_run": fast_forward_disposition,
            "observed_maximum_forced_suffix_tokens": max_forced_run,
            "roadmap_minimum_local_run_tokens": FAST_FORWARD_MIN_RUN,
            "restricted_lm_head": row_disposition,
            "maximum_admissible_rows": max_rows,
            "maximum_admissible_row_fraction": max_row_fraction,
            "minimum_topological_row_pruning_fraction": 1.0 - max_row_fraction,
            "restricted_lm_head_row_limit": row_screen_limit,
            "restricted_lm_head_missing_evidence": [
                "quantized block bytes touched",
                "metadata traffic",
                "restricted dispatch cost",
                "exact selected-argmax implementation",
                "whole-token projection",
            ],
            "authority": "scoped forced-run KILL and row-topology signal only",
        }),
        claim_boundary: json!({
            "clean_assistant_output_boundary": true,
            "token_healing": false,
            "piece_decoding": "exact_concatenated_bytes",
            "grammar_mask_scope": "full_vocabulary_before_sampling",
            "additional_logit_masks_in_forcedness": false,
            "terminal_action": "separate_from_content_rows",
            "scope": "exact finite language and fingerprinted token-piece policy only",
            "canonical_path_fingerprint_scope": "these 36 target-tokenizer retokenizations",
        }),
        inputs: json!({
            "model_path": args.model,
            "grammar_path": args.grammar,
            "grammar_file_sha256": sha256_hex(grammar_bytes),
            "trace_path": args.traces,
            "trace_file_sha256": sha256_hex(trace_bytes),
        }),
        tokenizer: json!({
            "vocab_rows": vocab_rows,
            "stop_token_ids": stop_token_ids,
            "admissible_ordinary_nonempty_rows": inventory.admissible_nonempty_rows,
            "ordinary_content_empty_rows": inventory.ordinary_content_empty_rows,
            "excluded_noncontent_nonempty_rows": inventory.excluded_noncontent_nonempty_rows,
            "excluded_noncontent_empty_rows": inventory.excluded_noncontent_empty_rows,
            "inventory_counts_are_disjoint": true,
            "duplicate_eligible_piece_groups": inventory.duplicate_piece_groups,
            "max_token_ids_per_eligible_piece": inventory.max_ids_per_piece,
            "grammar_piece_policy_sha256": inventory.grammar_piece_policy_sha256,
        }),
        grammar: json!({
            "spec": grammar,
            "language_strings": members.len(),
            "prefix_free": true,
            "finite_language_sha256": language_digest(members),
        }),
        global_topology: json!({
            "productive_prefix_states": topology.states.len(),
            "reachable_prefix_states": reachable.len(),
            "unreachable_prefix_states": topology.states.len() - reachable.len(),
            "terminal_states": reachable.iter().filter(|state| state.terminal).count(),
            "nonterminal_states": nonterminal.len(),
            "singleton_states": nonterminal
                .iter()
                .filter(|state| state.admissible.len() == 1)
                .count(),
            "branch_states": branch_rows.len(),
            "state_token_incidences": state_token_incidences,
            "branch_state_token_incidences": branch_state_token_incidences,
            "unique_admissible_token_rows": unique_admissible_tokens.len(),
            "unique_admissible_token_row_fraction":
                unique_admissible_tokens.len() as f64 / vocab_rows as f64,
            "unique_admissible_token_rows_sha256":
                token_id_set_digest(&unique_admissible_tokens),
            "unique_branch_token_rows": unique_branch_tokens.len(),
            "unique_branch_token_row_fraction":
                unique_branch_tokens.len() as f64 / vocab_rows as f64,
            "unique_branch_token_rows_sha256": token_id_set_digest(&unique_branch_tokens),
            "maximum_forced_suffix_tokens": max_forced_run,
            "states_attaining_maximum_forced_suffix": {
                "count": states_attaining_maximum.len(),
                "sha256": state_subset_digest(&states_attaining_maximum),
                "examples": maximum_state_examples,
            },
            "admissible_row_count_histogram": histogram(global_rows.iter().copied()),
            "admissible_row_count_quantiles": quantiles(&global_rows),
            "branch_admissible_row_count_quantiles": quantiles(&branch_rows),
            "maximum_admissible_row_fraction": max_row_fraction,
            "minimum_topological_row_pruning_fraction": 1.0 - max_row_fraction,
            "indexed_naive_cross_checks": topology.indexed_naive_cross_checks,
            "state_sha256": topology.state_sha256,
        }),
        canonical_paths: json!({
            "path_count": canonical_paths.len(),
            "canonical_path_sha256": canonical_path_digest(canonical_paths),
            "token_count_histogram": histogram(canonical_token_counts),
            "admissible_row_count_quantiles": quantiles(&canonical_rows),
            "forced_run_histogram": histogram(canonical_runs),
            "forced_run_count": canonical_forced_run_count,
            "maximum_forced_run": canonical_maximum_forced_run,
            "forced_token_positions": canonical_forced_token_positions,
        }),
        real_trace_weighting: json!({
            "fixture_id": traces.fixture_id,
            "source": traces.source,
            "interpretation": "real source-model output bytes retokenized canonically by target tokenizer",
            "not_authority_for": [
                "target-model sampled token paths",
                "broad structured-output trigger rate",
                "product speedup",
            ],
            "pooled": summarize_trace_rows(&all_records, &paths_by_text)?,
            "strata": trace_strata,
            "strata_are_not_broad_request_archetypes": true,
        }),
        counterfactual_fixed_n8_screen: json!({
            "serial_transition_ms": args.serial_transition_ms,
            "fixed_n8_packet_ms": args.fixed_n8_packet_ms,
            "run_tokens": max_forced_run,
            "cost_scope": "observed-max counterfactual only; no grammar scan or measured Cpack(r)",
            "cursor_definitions": {
                "aligned_nonterminal": "r transitions",
                "aligned_direct_terminal": "r-1 transitions",
                "leading_pending_terminal": "r transitions",
                "leading_pending_nonterminal": "r+1 transitions",
            },
            "cursor_cases": cursor_rows,
            "all_observed_max_cases_negative": cursor_cases.iter().all(|(_, case)| {
                required_transitions(max_forced_run, *case) as f64
                    * args.serial_transition_ms
                    - args.fixed_n8_packet_ms
                    < 0.0
            }),
        }),
    })
}

fn run(args: &Args) -> Result<OutputDocument> {
    ensure!(
        args.serial_transition_ms.is_finite() && args.serial_transition_ms > 0.0,
        "--serial-transition-ms must be finite and positive"
    );
    ensure!(
        args.fixed_n8_packet_ms.is_finite() && args.fixed_n8_packet_ms > 0.0,
        "--fixed-n8-packet-ms must be finite and positive"
    );
    let (grammar, grammar_bytes): (GrammarSpec, _) = read_json(&args.grammar)?;
    let (traces, trace_bytes): (TraceFixture, _) = read_json(&args.traces)?;
    let members = generate_language(&grammar)?;
    validate_trace_fixture(&traces, &members)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open tokenizer metadata from {:?}", args.model))?;
    let stop_token_ids = gguf
        .stop_token_ids()
        .context("resolve declared stop tokens")?;
    let tokenizer = NativeTokenizer::from_gguf(&gguf).context("construct native tokenizer")?;
    let inventory = build_token_inventory(&tokenizer)?;
    for &stop_token in &stop_token_ids {
        ensure!(
            inventory.index.piece(stop_token).is_none(),
            "declared stop token {stop_token} is admitted as ordinary content"
        );
    }
    let topology = analyze_topology(&members, &inventory.index)?;
    let canonical_paths =
        analyze_canonical_paths(&members, &tokenizer, &inventory.index, &topology)?;
    build_output(
        args,
        &grammar,
        &grammar_bytes,
        &traces,
        &trace_bytes,
        &stop_token_ids,
        &inventory,
        &members,
        &topology,
        &canonical_paths,
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output = run(&args)?;
    match &args.output {
        Some(path) => {
            let file = File::create(path).with_context(|| format!("create {path:?}"))?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &output)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        None => {
            let stdout = std::io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            serde_json::to_writer_pretty(&mut writer, &output)?;
            writer.write_all(b"\n")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(text: &str) -> LanguageMember {
        LanguageMember {
            text: text.to_string(),
            bytes: text.as_bytes().to_vec(),
            choices: Vec::new(),
        }
    }

    fn index(pieces: &[(i32, &str)]) -> TokenIndex {
        let vocab_size = pieces
            .iter()
            .map(|(id, _)| *id as usize + 1)
            .max()
            .unwrap_or(0);
        TokenIndex::new(
            vocab_size,
            pieces
                .iter()
                .map(|(id, text)| TokenPiece {
                    id: *id,
                    bytes: text.as_bytes().to_vec(),
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn ordered_enum_language_is_exact_and_prefix_free() {
        let spec = GrammarSpec {
            schema_version: 1,
            grammar_id: "tiny".into(),
            kind: "ordered_enum_object".into(),
            fields: vec![
                EnumField {
                    name: "a".into(),
                    values: vec!["x".into(), "y".into()],
                },
                EnumField {
                    name: "b".into(),
                    values: vec!["1".into(), "2".into()],
                },
            ],
        };
        let members = generate_language(&spec).unwrap();
        let texts: BTreeSet<_> = members.iter().map(|member| member.text.as_str()).collect();
        assert_eq!(members.len(), 4);
        assert!(texts.contains("{\"a\":\"x\",\"b\":\"1\"}"));
        assert!(texts.contains("{\"a\":\"y\",\"b\":\"2\"}"));
    }

    #[test]
    fn duplicate_piece_ids_prevent_false_singleton() {
        let members = vec![member("ab")];
        let index = index(&[(0, "a"), (1, "a"), (2, "b"), (3, "ab")]);
        let topology = analyze_topology(&members, &index).unwrap();
        let root = &topology.states[topology.by_prefix[&Vec::new()]];
        assert_eq!(root.admissible, vec![0, 1, 3]);
        let a = &topology.states[topology.by_prefix[b"a".as_slice()]];
        assert_eq!(a.admissible, vec![2]);
        assert_eq!(a.forced_suffix_len, 1);
    }

    #[test]
    fn noncanonical_tokenizations_converge_by_byte_prefix() {
        let members = vec![member("abc")];
        let index = index(&[
            (0, "a"),
            (1, "ab"),
            (2, "abc"),
            (3, "b"),
            (4, "bc"),
            (5, "c"),
            (6, "abcd"),
        ]);
        let topology = analyze_topology(&members, &index).unwrap();
        let root = &topology.states[topology.by_prefix[&Vec::new()]];
        assert_eq!(root.admissible, vec![0, 1, 2]);
        let a = &topology.states[topology.by_prefix[b"a".as_slice()]];
        assert_eq!(a.admissible, vec![3, 4]);
        let ab = &topology.states[topology.by_prefix[b"ab".as_slice()]];
        assert_eq!(ab.admissible, vec![5]);
        let terminal = &topology.states[topology.by_prefix[b"abc".as_slice()]];
        assert!(terminal.terminal);
        assert!(terminal.admissible.is_empty());
        assert!(!topology.by_prefix.contains_key(b"abcd".as_slice()));
    }

    #[test]
    fn forced_suffix_recurrence_counts_a_four_token_chain() {
        let members = vec![member("abcd")];
        let index = index(&[(0, "a"), (1, "b"), (2, "c"), (3, "d")]);
        let topology = analyze_topology(&members, &index).unwrap();
        let root = &topology.states[topology.by_prefix[&Vec::new()]];
        assert_eq!(root.forced_suffix_len, 4);
        let ab = &topology.states[topology.by_prefix[b"ab".as_slice()]];
        assert_eq!(ab.forced_suffix_len, 2);
    }

    #[test]
    fn productive_byte_prefix_can_be_token_unreachable() {
        let members = vec![member("ab")];
        let index = index(&[(0, "ab")]);
        let topology = analyze_topology(&members, &index).unwrap();
        assert!(topology.states[topology.by_prefix[&Vec::new()]].reachable);
        assert!(!topology.states[topology.by_prefix[b"a".as_slice()]].reachable);
        assert!(topology.states[topology.by_prefix[b"ab".as_slice()]].reachable);
    }

    #[test]
    fn canonical_run_extraction_respects_surrounding_branches() {
        let members = vec![
            member("xabq"),
            member("xabr"),
            member("yabq"),
            member("yabr"),
        ];
        let index = index(&[(0, "x"), (1, "y"), (2, "a"), (3, "b"), (4, "q"), (5, "r")]);
        let topology = analyze_topology(&members, &index).unwrap();
        let path = analyze_token_path(&members[0], vec![0, 2, 3, 4], &index, &topology)
            .expect("analyze canonical path");
        assert_eq!(path.admissible_rows_before_token, vec![2, 1, 1, 2]);
        assert_eq!(path.forced_runs, vec![2]);
    }

    #[test]
    fn cursor_transition_accounting_covers_all_four_cases() {
        assert_eq!(required_transitions(4, CursorCase::AlignedNonterminal), 4);
        assert_eq!(
            required_transitions(4, CursorCase::AlignedDirectTerminal),
            3
        );
        assert_eq!(
            required_transitions(4, CursorCase::LeadingPendingTerminal),
            4
        );
        assert_eq!(
            required_transitions(4, CursorCase::LeadingPendingNonterminal),
            5
        );
    }

    #[test]
    fn empty_piece_is_rejected_before_graph_construction() {
        let error = TokenIndex::new(
            1,
            vec![TokenPiece {
                id: 0,
                bytes: Vec::new(),
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("empty pieces"));
    }

    #[test]
    fn nearest_rank_quantiles_are_explicit() {
        let values = [1, 2, 3, 4, 5];
        assert_eq!(nearest_rank(&values, 50, 100), Some(3));
        assert_eq!(nearest_rank(&values, 90, 100), Some(5));
        assert_eq!(nearest_rank(&[], 50, 100), None);
    }
}
