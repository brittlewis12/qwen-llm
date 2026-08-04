use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputePipelineState,
};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::{Buffer, KernelEncoder, MetalContext, MetalTensor, encode_mat_vec_q6_k_f32},
    model::ArchKind,
    sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig, SamplingError},
    tensor::GgmlType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const RESULT_SCHEMA: &str = "qwen-grammar-lm-head-row-floor/v1";
const MANIFEST_SCHEMA: &str = "grammar-lm-head-row-banks/v1";
const MANIFEST_SHA256: &str = "2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0";
const MANIFEST_CLAIM: &str =
    "bench-only exact branch topology; no grammar runtime or performance authority";
const GRAMMAR_ID: &str = "response-shape-v1";
const VOCAB_ROWS: usize = 248_320;
const Q6_BLOCK_ELEMENTS: usize = 256;
const Q6_BLOCK_BYTES: usize = 210;
const BANK_ALIGNMENT: usize = 32;
const BANK_POISON: u8 = 0xa5;
const OUTPUT_GUARD_ELEMENTS: usize = 64;
const OUTPUT_GUARD_VALUE: f32 = -1234.5;
const OUTPUT_POISON_BITS: u32 = 0x7fc0_1234;
const WIDTHS: [usize; 9] = [2, 3, 4, 5, 6, 7, 11, 12, 17];
const HIDDEN_SEEDS: [u64; 4] = [1, 2, 3, 4];

const GRAMMAR_FILE_SHA256: &str =
    "f3a053733738aa8e05c10c2f078ca5a36c26a3bc5746487903463cdcf31b2634";
const TRACE_FILE_SHA256: &str = "eb702b6b93fe58255cc16783363150f886dbfd0d6f7c032c1798bae1c41b74b1";
const PIECE_POLICY_SHA256: &str =
    "58748dfe811c1710c732239ae59053f2888bf32c1c3ee4f8ec10f0ac089fdd60";
const FINITE_LANGUAGE_SHA256: &str =
    "94dc537ecb3d9209cfbc8e28b502916b20ee7b685393b6cf85ab610c08be0faf";
const STATE_SHA256: &str = "afa64a9a91677309fffbc54833b73d8881d7c4745127ae5ec10eebc87d70be40";
const BRANCH_STATE_SHA256: &str =
    "47b4cfc1608481f924d1cf7dec9887797da2cd7dd3c38bdd302b8340db68deb4";
const UNIQUE_ROWS_SHA256: &str = "c5fd62c703b9fb32fb9b505963754063a43ee827cdcff0d0e3e1c7cd8d9b763a";
const CANONICAL_PATH_SHA256: &str =
    "b72213294b55231dd4eaa12a05276be7dfe4e5ce994ff7a9451c258876333401";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum GrammarLmHeadProfile {
    A3b,
    Dense,
}

impl GrammarLmHeadProfile {
    fn spec(self) -> ProfileSpec {
        match self {
            Self::A3b => ProfileSpec {
                id: "a3b",
                expected_path: "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                file_bytes: 22_134_528_992,
                model_sha256: "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
                arch_kind: ArchKind::Moe,
                layers: 40,
                hidden: 2_048,
                head_bytes: 417_177_600,
                row_bytes: 1_680,
                bank_payload_bytes: 2_466_240,
                bank_padding_bytes: 2_720,
                bank_span_bytes: 2_468_960,
                t0_ms: 9.32,
            },
            Self::Dense => ProfileSpec {
                id: "dense",
                expected_path: "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
                file_bytes: 16_817_244_384,
                model_sha256: "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0",
                arch_kind: ArchKind::Dense,
                layers: 64,
                hidden: 5_120,
                head_bytes: 1_042_944_000,
                row_bytes: 4_200,
                bank_payload_bytes: 6_165_600,
                bank_padding_bytes: 5_408,
                bank_span_bytes: 6_171_008,
                t0_ms: 38.619,
            },
        }
    }
}

#[derive(Parser, Debug)]
pub struct GrammarLmHeadRowFloorArgs {
    /// Exact frozen model profile to authenticate and measure.
    #[arg(long, value_enum)]
    profile: GrammarLmHeadProfile,
    /// Exact GGUF path frozen for the selected profile.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Frozen v0.655 branch-bank manifest.
    #[arg(long)]
    manifest: PathBuf,
    /// Prior A3B GO result required before the dense guard may run.
    #[arg(long)]
    a3b_result: Option<PathBuf>,
    /// Complete unscored arm-by-width warmup rounds.
    #[arg(long, default_value = "5")]
    warmup_rounds: usize,
    /// Full-control-only screening rounds per width.
    #[arg(long, default_value = "12")]
    screen_rounds: usize,
    /// Counterbalanced paired rounds per width.
    #[arg(long, default_value = "12")]
    paired_rounds: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProfileSpec {
    id: &'static str,
    expected_path: &'static str,
    file_bytes: u64,
    model_sha256: &'static str,
    arch_kind: ArchKind,
    layers: u32,
    hidden: usize,
    head_bytes: usize,
    row_bytes: usize,
    bank_payload_bytes: usize,
    bank_padding_bytes: usize,
    bank_span_bytes: usize,
    t0_ms: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestDocument {
    schema: String,
    claim_scope: String,
    grammar_id: String,
    fingerprints: ManifestFingerprints,
    counts: ManifestCounts,
    row_count_histogram: BTreeMap<String, usize>,
    representative_bank_state_by_width: BTreeMap<String, usize>,
    states: Vec<ManifestState>,
    canonical_width_histogram: BTreeMap<String, usize>,
    canonical_paths: Vec<ManifestCanonicalPath>,
    trace_width_histogram: BTreeMap<String, usize>,
    trace_stratum_width_histograms: BTreeMap<String, BTreeMap<String, usize>>,
    trace_records: Vec<ManifestTraceRecord>,
    trace_source: ManifestTraceSource,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestFingerprints {
    branch_state_sha256: String,
    canonical_path_sha256: String,
    finite_language_sha256: String,
    grammar_file_sha256: String,
    grammar_piece_policy_sha256: String,
    state_sha256: String,
    trace_file_sha256: String,
    unique_branch_token_rows_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestCounts {
    branch_state_token_incidences: usize,
    branch_states: usize,
    canonical_branch_incidences: usize,
    canonical_paths: usize,
    productive_states: usize,
    trace_branch_incidences: usize,
    trace_records: usize,
    unique_branch_token_rows: usize,
    vocab_rows: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestState {
    bank_state_index: usize,
    topology_state_index: usize,
    prefix_bytes: usize,
    prefix_hex: String,
    prefix_sha256: String,
    token_ids: Vec<i32>,
    token_ids_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestChoice {
    field: String,
    value: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestCanonicalPath {
    path_index: usize,
    text_sha256: String,
    choices: Vec<ManifestChoice>,
    token_ids: Vec<i32>,
    bank_state_indices: Vec<usize>,
    topology_state_indices: Vec<usize>,
    widths: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestTraceRecord {
    item_id: String,
    branch: String,
    canonical_path_index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestTraceSource {
    source_model: String,
    results_path: String,
    results_sha256: String,
    prompt_builder_path: String,
    prompt_builder_sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedState {
    bank_state_index: usize,
    topology_state_index: usize,
    prefix: Vec<u8>,
    token_ids: Vec<i32>,
}

#[derive(Clone, Debug)]
struct ValidatedPath {
    path_index: usize,
    widths: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct ValidatedManifest {
    states: Vec<ValidatedState>,
    representative_by_width: BTreeMap<usize, usize>,
    row_histogram: BTreeMap<usize, usize>,
    canonical_histogram: BTreeMap<usize, usize>,
    canonical_paths: Vec<ValidatedPath>,
    trace_histogram: BTreeMap<usize, usize>,
    trace_stratum_histograms: BTreeMap<String, BTreeMap<usize, usize>>,
    fingerprints: ManifestFingerprints,
    counts: ManifestCounts,
    trace_source: ManifestTraceSource,
}

#[derive(Clone, Debug)]
struct StateLayout {
    state: ValidatedState,
    offset: usize,
    payload_bytes: usize,
    span_bytes: usize,
}

#[derive(Debug)]
struct BankLayout {
    states: Vec<StateLayout>,
    payload_bytes: usize,
    padding_bytes: usize,
    span_bytes: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn parse_hex(value: &str) -> Result<Vec<u8>> {
    ensure!(value.len().is_multiple_of(2), "hex string has odd length");
    ensure!(
        value.bytes().all(|byte| byte.is_ascii_hexdigit())
            && value.bytes().all(|byte| !byte.is_ascii_uppercase()),
        "hex string is not canonical lowercase hex"
    );
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("hex digits are ASCII");
            u8::from_str_radix(text, 16).context("decode hex byte")
        })
        .collect()
}

fn parse_histogram(raw: &BTreeMap<String, usize>) -> Result<BTreeMap<usize, usize>> {
    raw.iter()
        .map(|(key, &value)| {
            let width = key
                .parse::<usize>()
                .with_context(|| format!("invalid width key {key:?}"))?;
            ensure!(width.to_string() == *key, "noncanonical width key {key:?}");
            Ok((width, value))
        })
        .collect()
}

fn histogram(values: impl IntoIterator<Item = usize>) -> BTreeMap<usize, usize> {
    let mut histogram = BTreeMap::new();
    for value in values {
        *histogram.entry(value).or_insert(0) += 1;
    }
    histogram
}

fn token_slice_digest(tokens: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-token-id-slice/v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn token_set_digest(tokens: &BTreeSet<i32>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-token-id-set/v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn branch_state_digest(states: &[ValidatedState]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-state-subset/v1\0");
    for state in states {
        hasher.update((state.prefix.len() as u64).to_le_bytes());
        hasher.update(&state.prefix);
        hasher.update(0u64.to_le_bytes());
        hasher.update((state.token_ids.len() as u64).to_le_bytes());
        for token in &state.token_ids {
            hasher.update(token.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn expected_row_histogram() -> BTreeMap<usize, usize> {
    BTreeMap::from([
        (2, 207),
        (3, 123),
        (4, 55),
        (5, 38),
        (6, 7),
        (7, 4),
        (11, 4),
        (12, 12),
        (17, 1),
    ])
}

fn expected_canonical_histogram() -> BTreeMap<usize, usize> {
    BTreeMap::from([
        (2, 108),
        (3, 180),
        (4, 63),
        (5, 144),
        (6, 45),
        (11, 36),
        (12, 36),
        (17, 36),
    ])
}

fn expected_trace_histogram() -> BTreeMap<usize, usize> {
    BTreeMap::from([
        (2, 60),
        (3, 100),
        (4, 40),
        (5, 80),
        (6, 20),
        (11, 20),
        (12, 20),
        (17, 20),
    ])
}

pub(crate) fn validate_manifest(document: ManifestDocument) -> Result<ValidatedManifest> {
    ensure!(
        document.schema == MANIFEST_SCHEMA,
        "manifest schema mismatch"
    );
    ensure!(
        document.claim_scope == MANIFEST_CLAIM,
        "manifest claim scope mismatch"
    );
    ensure!(
        document.grammar_id == GRAMMAR_ID,
        "manifest grammar mismatch"
    );
    let fingerprints = &document.fingerprints;
    for digest in [
        &fingerprints.branch_state_sha256,
        &fingerprints.canonical_path_sha256,
        &fingerprints.finite_language_sha256,
        &fingerprints.grammar_file_sha256,
        &fingerprints.grammar_piece_policy_sha256,
        &fingerprints.state_sha256,
        &fingerprints.trace_file_sha256,
        &fingerprints.unique_branch_token_rows_sha256,
    ] {
        ensure!(valid_sha256(digest), "manifest contains an invalid SHA-256");
    }
    ensure!(
        fingerprints.grammar_file_sha256 == GRAMMAR_FILE_SHA256
            && fingerprints.trace_file_sha256 == TRACE_FILE_SHA256
            && fingerprints.grammar_piece_policy_sha256 == PIECE_POLICY_SHA256
            && fingerprints.finite_language_sha256 == FINITE_LANGUAGE_SHA256
            && fingerprints.state_sha256 == STATE_SHA256
            && fingerprints.branch_state_sha256 == BRANCH_STATE_SHA256
            && fingerprints.unique_branch_token_rows_sha256 == UNIQUE_ROWS_SHA256
            && fingerprints.canonical_path_sha256 == CANONICAL_PATH_SHA256,
        "manifest fingerprint mismatch"
    );
    let counts = &document.counts;
    ensure!(
        counts.vocab_rows == VOCAB_ROWS,
        "manifest vocabulary mismatch"
    );
    ensure!(counts.productive_states == 616, "productive-state mismatch");
    ensure!(counts.branch_states == 451, "branch-state mismatch");
    ensure!(
        counts.branch_state_token_incidences == 1_468,
        "branch-incidence mismatch"
    );
    ensure!(
        counts.unique_branch_token_rows == 222,
        "unique-row mismatch"
    );
    ensure!(counts.canonical_paths == 36, "canonical-path mismatch");
    ensure!(
        counts.canonical_branch_incidences == 648,
        "canonical-incidence mismatch"
    );
    ensure!(counts.trace_records == 20, "trace-record mismatch");
    ensure!(
        counts.trace_branch_incidences == 360,
        "trace-incidence mismatch"
    );

    ensure!(
        document.states.len() == 451,
        "manifest must contain 451 states"
    );
    let mut states = Vec::with_capacity(document.states.len());
    let mut previous_prefix: Option<Vec<u8>> = None;
    let mut previous_topology_index = None;
    let mut unique_rows = BTreeSet::new();
    for (position, state) in document.states.into_iter().enumerate() {
        ensure!(
            state.bank_state_index == position,
            "bank state index {} does not match position {position}",
            state.bank_state_index
        );
        if let Some(previous) = previous_topology_index {
            ensure!(
                state.topology_state_index > previous,
                "topology state indices are not strictly increasing"
            );
        }
        ensure!(
            state.topology_state_index < counts.productive_states,
            "topology state index is out of range"
        );
        let prefix = parse_hex(&state.prefix_hex)?;
        ensure!(prefix.len() == state.prefix_bytes, "prefix length mismatch");
        ensure!(
            sha256_hex(&prefix) == state.prefix_sha256,
            "prefix hash mismatch"
        );
        if let Some(previous) = &previous_prefix {
            ensure!(
                prefix > *previous,
                "branch prefixes are not strictly ordered"
            );
        }
        ensure!(
            WIDTHS.contains(&state.token_ids.len()),
            "unsupported state width {}",
            state.token_ids.len()
        );
        ensure!(
            state.token_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "state token IDs are not strictly increasing"
        );
        ensure!(
            state
                .token_ids
                .iter()
                .all(|&token| token >= 0 && token < VOCAB_ROWS as i32),
            "state token ID is out of range"
        );
        ensure!(
            token_slice_digest(&state.token_ids) == state.token_ids_sha256,
            "state token digest mismatch"
        );
        unique_rows.extend(state.token_ids.iter().copied());
        previous_topology_index = Some(state.topology_state_index);
        previous_prefix = Some(prefix.clone());
        states.push(ValidatedState {
            bank_state_index: state.bank_state_index,
            topology_state_index: state.topology_state_index,
            prefix,
            token_ids: state.token_ids,
        });
    }
    ensure!(unique_rows.len() == 222, "unique row count mismatch");
    ensure!(
        token_set_digest(&unique_rows) == UNIQUE_ROWS_SHA256,
        "unique row digest mismatch"
    );
    ensure!(
        branch_state_digest(&states) == BRANCH_STATE_SHA256,
        "branch-state digest mismatch"
    );
    let row_histogram = histogram(states.iter().map(|state| state.token_ids.len()));
    ensure!(
        row_histogram == expected_row_histogram(),
        "state width histogram mismatch"
    );
    ensure!(
        row_histogram == parse_histogram(&document.row_count_histogram)?,
        "reported state width histogram mismatch"
    );
    ensure!(
        states
            .iter()
            .map(|state| state.token_ids.len())
            .sum::<usize>()
            == 1_468,
        "state incidence total mismatch"
    );

    let reported_representatives = parse_histogram(&document.representative_bank_state_by_width)?;
    let mut representative_by_width = BTreeMap::new();
    for state in &states {
        representative_by_width
            .entry(state.token_ids.len())
            .or_insert(state.bank_state_index);
    }
    ensure!(
        representative_by_width == reported_representatives,
        "representative state map mismatch"
    );

    ensure!(
        document.canonical_paths.len() == 36,
        "manifest must contain 36 canonical paths"
    );
    let mut canonical_widths = Vec::new();
    let mut canonical_paths = Vec::with_capacity(36);
    let mut path_hashes = BTreeSet::new();
    for (position, path) in document.canonical_paths.into_iter().enumerate() {
        ensure!(path.path_index == position, "canonical path index mismatch");
        ensure!(
            valid_sha256(&path.text_sha256) && path_hashes.insert(path.text_sha256),
            "duplicate or invalid canonical text hash"
        );
        ensure!(
            path.choices.len() == 3
                && path
                    .choices
                    .iter()
                    .all(|choice| !choice.field.is_empty() && !choice.value.is_empty()),
            "canonical path choices are malformed"
        );
        ensure!(
            path.token_ids.len() == 18
                && path.bank_state_indices.len() == 18
                && path.topology_state_indices.len() == 18
                && path.widths.len() == 18,
            "canonical path arrays must all contain 18 entries"
        );
        for index in 0..18 {
            let bank_index = path.bank_state_indices[index];
            let state = states
                .get(bank_index)
                .context("canonical bank state index is out of range")?;
            ensure!(
                state.topology_state_index == path.topology_state_indices[index],
                "canonical topology index mismatch"
            );
            ensure!(
                state.token_ids.len() == path.widths[index],
                "canonical width mismatch"
            );
            ensure!(
                state
                    .token_ids
                    .binary_search(&path.token_ids[index])
                    .is_ok(),
                "canonical token is absent from referenced state"
            );
        }
        canonical_widths.extend_from_slice(&path.widths);
        canonical_paths.push(ValidatedPath {
            path_index: path.path_index,
            widths: path.widths,
        });
    }
    let canonical_histogram = histogram(canonical_widths);
    ensure!(
        canonical_histogram == expected_canonical_histogram(),
        "canonical width histogram mismatch"
    );
    ensure!(
        canonical_histogram == parse_histogram(&document.canonical_width_histogram)?,
        "reported canonical width histogram mismatch"
    );

    ensure!(
        document.trace_records.len() == 20,
        "manifest must contain 20 trace records"
    );
    let mut trace_keys = BTreeSet::new();
    let mut trace_widths = Vec::new();
    let mut trace_stratum_widths: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for record in document.trace_records {
        ensure!(
            record.branch == "Q-self-pred" || record.branch == "Q-other-pred",
            "unexpected trace stratum"
        );
        ensure!(
            trace_keys.insert((record.item_id.clone(), record.branch.clone())),
            "duplicate trace record identity"
        );
        let path = canonical_paths
            .get(record.canonical_path_index)
            .context("trace canonical path index is out of range")?;
        trace_widths.extend_from_slice(&path.widths);
        trace_stratum_widths
            .entry(record.branch)
            .or_default()
            .extend_from_slice(&path.widths);
    }
    let trace_histogram = histogram(trace_widths);
    ensure!(
        trace_histogram == expected_trace_histogram(),
        "trace width histogram mismatch"
    );
    ensure!(
        trace_histogram == parse_histogram(&document.trace_width_histogram)?,
        "reported trace width histogram mismatch"
    );
    let trace_stratum_histograms: BTreeMap<_, _> = trace_stratum_widths
        .into_iter()
        .map(|(stratum, widths)| (stratum, histogram(widths)))
        .collect();
    let reported_strata: BTreeMap<_, _> = document
        .trace_stratum_width_histograms
        .iter()
        .map(|(stratum, histogram)| Ok((stratum.clone(), parse_histogram(histogram)?)))
        .collect::<Result<_>>()?;
    ensure!(
        trace_stratum_histograms == reported_strata,
        "trace stratum histogram mismatch"
    );
    ensure!(
        trace_stratum_histograms.len() == 2
            && trace_stratum_histograms
                .values()
                .all(|histogram| histogram.values().sum::<usize>() == 180),
        "trace stratum incidence mismatch"
    );

    let source = &document.trace_source;
    ensure!(
        source.source_model == "Qwen/Qwen3-4B"
            && source.results_path
                == "self_trajectory/spike_rf/analysis/cache/spike_rf/results.json"
            && source.results_sha256
                == "4a6de539fbad9cfe89f8210fa3aff542137b4b6bb15ab95dacadafb891e8afcc"
            && source.prompt_builder_path == "self_traj_spike_rf.py"
            && source.prompt_builder_sha256
                == "1c7e3595a938aeb13676e85f4e7ff02431e667b3db466c8ced766f9c371a4740",
        "trace source identity mismatch"
    );

    Ok(ValidatedManifest {
        states,
        representative_by_width,
        row_histogram,
        canonical_histogram,
        canonical_paths,
        trace_histogram,
        trace_stratum_histograms,
        fingerprints: document.fingerprints,
        counts: document.counts,
        trace_source: document.trace_source,
    })
}

impl ValidatedState {
    pub(crate) fn bank_state_index(&self) -> usize {
        self.bank_state_index
    }

    pub(crate) fn topology_state_index(&self) -> usize {
        self.topology_state_index
    }

    pub(crate) fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    pub(crate) fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }
}

impl ValidatedManifest {
    pub(crate) fn states(&self) -> &[ValidatedState] {
        &self.states
    }
}

fn checked_align_up(value: usize, alignment: usize) -> Result<usize> {
    ensure!(
        alignment.is_power_of_two(),
        "alignment is not a power of two"
    );
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("alignment overflow")
}

fn build_bank_layout(manifest: &ValidatedManifest, profile: ProfileSpec) -> Result<BankLayout> {
    ensure!(
        profile.hidden > 0 && profile.hidden.is_multiple_of(Q6_BLOCK_ELEMENTS),
        "profile hidden size is not Q6_K row-aligned"
    );
    let row_bytes = profile
        .hidden
        .checked_div(Q6_BLOCK_ELEMENTS)
        .and_then(|blocks| blocks.checked_mul(Q6_BLOCK_BYTES))
        .context("Q6_K row-byte overflow")?;
    ensure!(row_bytes == profile.row_bytes, "profile row-byte mismatch");
    ensure!(
        row_bytes.checked_mul(VOCAB_ROWS) == Some(profile.head_bytes),
        "profile head-byte mismatch"
    );

    let mut states = Vec::with_capacity(manifest.states.len());
    let mut offset = 0usize;
    let mut payload_total = 0usize;
    for state in &manifest.states {
        let payload_bytes = state
            .token_ids
            .len()
            .checked_mul(row_bytes)
            .context("state payload overflow")?;
        let span_bytes = checked_align_up(payload_bytes, BANK_ALIGNMENT)?;
        ensure!(
            offset.is_multiple_of(BANK_ALIGNMENT),
            "bank state offset is misaligned"
        );
        states.push(StateLayout {
            state: state.clone(),
            offset,
            payload_bytes,
            span_bytes,
        });
        payload_total = payload_total
            .checked_add(payload_bytes)
            .context("bank payload total overflow")?;
        offset = offset
            .checked_add(span_bytes)
            .context("bank span overflow")?;
    }
    let padding_bytes = offset
        .checked_sub(payload_total)
        .context("bank padding underflow")?;
    ensure!(
        payload_total == profile.bank_payload_bytes
            && padding_bytes == profile.bank_padding_bytes
            && offset == profile.bank_span_bytes,
        "bank totals do not match the frozen profile"
    );
    Ok(BankLayout {
        states,
        payload_bytes: payload_total,
        padding_bytes,
        span_bytes: offset,
    })
}

#[derive(Debug)]
struct PreparedBank {
    bytes: Vec<u8>,
    source_row_hashes: BTreeMap<i32, String>,
    source_rows_sha256: String,
}

fn source_row_set_digest(rows: &BTreeMap<i32, String>) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-source-row-set/v1\0");
    for (&token, digest) in rows {
        let raw = parse_hex(digest)?;
        ensure!(raw.len() == 32, "source row digest is not 32 bytes");
        hasher.update(token.to_le_bytes());
        hasher.update((raw.len() as u64).to_le_bytes());
        hasher.update(raw);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn prepare_bank(layout: &BankLayout, source: &[u8], profile: ProfileSpec) -> Result<PreparedBank> {
    ensure!(
        source.len() == profile.head_bytes,
        "head source size mismatch"
    );
    let mut bytes = vec![BANK_POISON; layout.span_bytes];
    let mut source_row_hashes = BTreeMap::new();
    for state in &layout.states {
        let mut destination = state.offset;
        for &token in &state.state.token_ids {
            let token = usize::try_from(token).context("negative source token")?;
            let source_start = token
                .checked_mul(profile.row_bytes)
                .context("source row offset overflow")?;
            let source_end = source_start
                .checked_add(profile.row_bytes)
                .context("source row endpoint overflow")?;
            let destination_end = destination
                .checked_add(profile.row_bytes)
                .context("destination row endpoint overflow")?;
            let source_row = source
                .get(source_start..source_end)
                .context("source row exceeds output.weight")?;
            let destination_row = bytes
                .get_mut(destination..destination_end)
                .context("destination row exceeds bank payload")?;
            destination_row.copy_from_slice(source_row);
            source_row_hashes
                .entry(token as i32)
                .or_insert_with(|| sha256_hex(source_row));
            destination = destination_end;
        }
        ensure!(
            destination == state.offset + state.payload_bytes,
            "state row copies do not cover the payload"
        );
        ensure!(
            bytes[state.offset + state.payload_bytes..state.offset + state.span_bytes]
                .iter()
                .all(|&byte| byte == BANK_POISON),
            "state padding poison changed during construction"
        );
    }
    let expected_source_rows: BTreeSet<_> = layout
        .states
        .iter()
        .flat_map(|state| state.state.token_ids.iter().copied())
        .collect();
    ensure!(
        source_row_hashes.len() == expected_source_rows.len()
            && source_row_hashes.keys().copied().eq(expected_source_rows),
        "source row hash set mismatch"
    );
    for state in &layout.states {
        for (local, &token) in state.state.token_ids.iter().enumerate() {
            let source_start = usize::try_from(token)?
                .checked_mul(profile.row_bytes)
                .context("source verification offset overflow")?;
            let destination_start = state
                .offset
                .checked_add(
                    local
                        .checked_mul(profile.row_bytes)
                        .context("destination verification offset overflow")?,
                )
                .context("destination verification endpoint overflow")?;
            ensure!(
                bytes[destination_start..destination_start + profile.row_bytes]
                    == source[source_start..source_start + profile.row_bytes],
                "bank row differs from source row"
            );
        }
    }
    let source_rows_sha256 = source_row_set_digest(&source_row_hashes)?;
    Ok(PreparedBank {
        bytes,
        source_row_hashes,
        source_rows_sha256,
    })
}

fn verify_source_rows(
    source: &[u8],
    profile: ProfileSpec,
    expected: &BTreeMap<i32, String>,
) -> Result<String> {
    let mut actual = BTreeMap::new();
    for &token in expected.keys() {
        let start = usize::try_from(token)?
            .checked_mul(profile.row_bytes)
            .context("source row verification offset overflow")?;
        let end = start
            .checked_add(profile.row_bytes)
            .context("source row verification endpoint overflow")?;
        let row = source
            .get(start..end)
            .context("source row verification exceeds output.weight")?;
        let digest = sha256_hex(row);
        ensure!(
            expected.get(&token) == Some(&digest),
            "source row {token} changed"
        );
        actual.insert(token, digest);
    }
    source_row_set_digest(&actual)
}

fn buffer_bytes(buffer: &Buffer, len: usize) -> Result<Vec<u8>> {
    ensure!(len <= buffer.length(), "buffer read exceeds allocation");
    let contents = buffer.contents();
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u8>(), len) };
    Ok(bytes.to_vec())
}

fn tensor_byte_range(tensor: &MetalTensor) -> Result<(usize, usize)> {
    let start = usize::try_from(tensor.offset).context("tensor offset does not fit usize")?;
    let len = usize::try_from(tensor.n_bytes()).context("tensor size does not fit usize")?;
    let end = start.checked_add(len).context("tensor endpoint overflow")?;
    ensure!(end <= tensor.buffer.length(), "tensor exceeds its buffer");
    Ok((start, end))
}

fn write_f32_tensor(tensor: &MetalTensor, values: &[f32]) -> Result<()> {
    ensure!(tensor.is_writable(), "host write requires writable tensor");
    ensure!(
        tensor.dtype == GgmlType::F32,
        "host write requires F32 tensor"
    );
    ensure!(
        tensor.n_elements() as usize == values.len(),
        "host write length mismatch"
    );
    let (start, end) = tensor_byte_range(tensor)?;
    ensure!(
        end - start == std::mem::size_of_val(values),
        "host write byte mismatch"
    );
    let contents = tensor.buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            contents.as_ptr().cast::<u8>().add(start),
            end - start,
        );
    }
    Ok(())
}

fn read_f32_tensor(tensor: &MetalTensor) -> Result<Vec<f32>> {
    ensure!(
        tensor.dtype == GgmlType::F32,
        "host read requires F32 tensor"
    );
    let count = usize::try_from(tensor.n_elements()).context("F32 element count overflow")?;
    let (start, end) = tensor_byte_range(tensor)?;
    ensure!(end - start == count * 4, "host read byte mismatch");
    let contents = tensor.buffer.contents();
    let values = unsafe {
        std::slice::from_raw_parts(
            contents.as_ptr().cast::<u8>().add(start).cast::<f32>(),
            count,
        )
    };
    Ok(values.to_vec())
}

fn write_compact_poison(storage: &MetalTensor, width: usize) -> Result<()> {
    let total = usize::try_from(storage.n_elements()).context("compact storage size overflow")?;
    ensure!(
        total >= 2 * OUTPUT_GUARD_ELEMENTS + width,
        "compact storage is too small"
    );
    let mut values = vec![OUTPUT_GUARD_VALUE; total];
    values[OUTPUT_GUARD_ELEMENTS..OUTPUT_GUARD_ELEMENTS + width]
        .fill(f32::from_bits(OUTPUT_POISON_BITS));
    write_f32_tensor(storage, &values)
}

fn validate_compact_guards(storage: &MetalTensor, width: usize) -> Result<()> {
    let values = read_f32_tensor(storage)?;
    let guard_bits = OUTPUT_GUARD_VALUE.to_bits();
    ensure!(
        values[..OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits),
        "compact prefix guard changed"
    );
    let suffix = OUTPUT_GUARD_ELEMENTS + width;
    ensure!(
        values[suffix..suffix + OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits),
        "compact suffix guard changed"
    );
    Ok(())
}

fn validate_compact_outer_guards(storage: &MetalTensor) -> Result<()> {
    let values = read_f32_tensor(storage)?;
    let guard_bits = OUTPUT_GUARD_VALUE.to_bits();
    ensure!(
        values[..OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits),
        "compact outer prefix guard changed"
    );
    let suffix = OUTPUT_GUARD_ELEMENTS + WIDTHS[WIDTHS.len() - 1];
    ensure!(
        values[suffix..suffix + OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits),
        "compact outer suffix guard changed"
    );
    Ok(())
}

fn hidden_vector(seed: u64, hidden: usize) -> Vec<f32> {
    let mut state = seed;
    (0..hidden)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((((state >> 40) & 0xffff) as i32) - 32_768) as f32 / 32_768.0
        })
        .collect()
}

fn validate_dispatch(
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    ensure!(n_in <= u32::MAX as usize && n_out <= u32::MAX as usize);
    ensure!(
        weight.dtype == GgmlType::Q6_K && weight.shape.as_slice() == [n_in as u64, n_out as u64],
        "Q6_K dispatch weight shape mismatch"
    );
    ensure!(
        x.dtype == GgmlType::F32 && x.shape.as_slice() == [n_in as u64],
        "Q6_K dispatch input shape mismatch"
    );
    ensure!(
        y.dtype == GgmlType::F32 && y.shape.as_slice() == [n_out as u64] && y.is_writable(),
        "Q6_K dispatch output shape mismatch"
    );
    tensor_byte_range(weight)?;
    tensor_byte_range(x)?;
    tensor_byte_range(y)?;
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct DispatchTiming {
    predecessor: String,
    command: &'static str,
    status: &'static str,
    error_none: bool,
    matvec_dispatches: usize,
    wall_ms: f64,
    gpu_ms: f64,
}

fn command_gpu_ms(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<f64> {
    let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    ensure!(
        gpu_ms.is_finite() && gpu_ms > 0.0,
        "invalid command-buffer GPU time {gpu_ms}"
    );
    Ok(gpu_ms)
}

fn check_command(cmd: &ProtocolObject<dyn MTLCommandBuffer>, label: &str) -> Result<()> {
    let status = cmd.status();
    let error = cmd.error();
    ensure!(
        status == MTLCommandBufferStatus::Completed && error.is_none(),
        "{label} failed: status={status:?} error={error:?}"
    );
    Ok(())
}

fn run_q6_dispatch(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    label: &str,
) -> Result<DispatchTiming> {
    validate_dispatch(weight, x, y, n_in, n_out)?;
    let start = Instant::now();
    let cmd = ctx
        .queue
        .commandBuffer()
        .with_context(|| format!("{label} command buffer"))?;
    let enc = KernelEncoder::begin(&cmd);
    encode_mat_vec_q6_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    check_command(&cmd, label)?;
    Ok(DispatchTiming {
        predecessor: label.to_string(),
        command: "kernel_mat_vec_q6_K_f32",
        status: "completed",
        error_none: true,
        matvec_dispatches: 1,
        wall_ms,
        gpu_ms: command_gpu_ms(&cmd)?,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MappedOutcome {
    Token {
        original_token: i32,
        candidate_index: usize,
        draws_before: usize,
        draws_after: usize,
    },
    Error {
        error: String,
        original_token: Option<i32>,
        draws_before: usize,
        draws_after: usize,
    },
}

fn all_negative_infinity(logits: &[f32]) -> bool {
    !logits.is_empty() && logits.iter().all(|&value| value == f32::NEG_INFINITY)
}

fn map_sampling_error(
    error: SamplingError,
    token_ids: Option<&[i32]>,
    draws_before: usize,
    draws_after: usize,
) -> Result<MappedOutcome> {
    let (error, original_token) = match error {
        SamplingError::NanLogit { token } => {
            let original_token = match token_ids {
                Some(ids) => Some(*ids.get(token).context("local NaN token is out of range")?),
                None => Some(i32::try_from(token).context("NaN token does not fit i32")?),
            };
            ("nan_logit".to_string(), original_token)
        }
        SamplingError::NoCandidates => ("no_candidates".to_string(), None),
        other => (other.to_string(), None),
    };
    Ok(MappedOutcome::Error {
        error,
        original_token,
        draws_before,
        draws_after,
    })
}

fn sample_compact_with_sampler(
    sampler: &mut Sampler,
    logits: &[f32],
    token_ids: &[i32],
) -> Result<MappedOutcome> {
    ensure!(
        logits.len() == token_ids.len(),
        "compact sampler length mismatch"
    );
    ensure!(
        token_ids.windows(2).all(|pair| pair[0] < pair[1]),
        "compact sampler token IDs are not strictly increasing"
    );
    let draws_before = sampler.draws();
    if all_negative_infinity(logits) {
        return Ok(MappedOutcome::Error {
            error: "all_negative_infinity".to_string(),
            original_token: None,
            draws_before,
            draws_after: sampler.draws(),
        });
    }
    match sampler.sample(logits) {
        Ok(sampled) => {
            let local = usize::try_from(sampled.token).context("negative local sampled token")?;
            let original_token = *token_ids
                .get(local)
                .context("local sampled token is out of range")?;
            Ok(MappedOutcome::Token {
                original_token,
                candidate_index: sampled.candidate_index,
                draws_before,
                draws_after: sampler.draws(),
            })
        }
        Err(error) => map_sampling_error(error, Some(token_ids), draws_before, sampler.draws()),
    }
}

fn sample_full_mask(
    config: SamplingConfig,
    compact_logits: &[f32],
    token_ids: &[i32],
) -> Result<MappedOutcome> {
    ensure!(compact_logits.len() == token_ids.len());
    let mut full = vec![f32::NEG_INFINITY; VOCAB_ROWS];
    for (&token, &logit) in token_ids.iter().zip(compact_logits) {
        full[usize::try_from(token)?] = logit;
    }
    let mut sampler = Sampler::new(config)?;
    let draws_before = sampler.draws();
    if all_negative_infinity(compact_logits) {
        return Ok(MappedOutcome::Error {
            error: "all_negative_infinity".to_string(),
            original_token: None,
            draws_before,
            draws_after: sampler.draws(),
        });
    }
    match sampler.sample(&full) {
        Ok(sampled) => Ok(MappedOutcome::Token {
            original_token: sampled.token,
            candidate_index: sampled.candidate_index,
            draws_before,
            draws_after: sampler.draws(),
        }),
        Err(error) => map_sampling_error(error, None, draws_before, sampler.draws()),
    }
}

fn assert_sampler_equivalence(
    config: SamplingConfig,
    logits: &[f32],
    token_ids: &[i32],
) -> Result<()> {
    let mut compact_sampler = Sampler::new(config)?;
    let compact = sample_compact_with_sampler(&mut compact_sampler, logits, token_ids)?;
    let full = sample_full_mask(config, logits, token_ids)?;
    ensure!(compact == full, "compact/full-mask sampler mismatch");
    Ok(())
}

fn validate_sampler_contract(manifest: &ValidatedManifest) -> Result<usize> {
    ensure!(SAMPLER_ALGORITHM_VERSION == 1, "sampler version changed");
    let state_index = *manifest
        .representative_by_width
        .get(&17)
        .context("missing width-17 sampler state")?;
    let token_ids = &manifest.states[state_index].token_ids;
    let mut cases = 0usize;

    let distinct: Vec<f32> = (0..token_ids.len())
        .map(|index| index as f32 / 4.0)
        .collect();
    assert_sampler_equivalence(SamplingConfig::default(), &distinct, token_ids)?;
    cases += 1;
    assert_sampler_equivalence(
        SamplingConfig::default(),
        &vec![1.0; token_ids.len()],
        token_ids,
    )?;
    cases += 1;
    for seed in [0, 1, 42, u64::MAX] {
        assert_sampler_equivalence(SamplingConfig::qwen_chat(seed), &distinct, token_ids)?;
        cases += 1;
    }
    assert_sampler_equivalence(
        SamplingConfig::qwen_chat(0),
        &vec![0.5; token_ids.len()],
        token_ids,
    )?;
    cases += 1;
    let mut infinities = distinct.clone();
    infinities[1] = f32::INFINITY;
    infinities[token_ids.len() - 2] = f32::INFINITY;
    assert_sampler_equivalence(SamplingConfig::qwen_chat(42), &infinities, token_ids)?;
    cases += 1;
    assert_sampler_equivalence(
        SamplingConfig {
            temperature: 0.7,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed: 1,
        },
        &distinct,
        token_ids,
    )?;
    cases += 1;
    for position in 0..token_ids.len() {
        let mut nan_case = distinct.clone();
        nan_case[position] = f32::from_bits(OUTPUT_POISON_BITS);
        assert_sampler_equivalence(SamplingConfig::qwen_chat(0), &nan_case, token_ids)?;
        cases += 1;
    }
    assert_sampler_equivalence(
        SamplingConfig::qwen_chat(0),
        &vec![f32::NEG_INFINITY; token_ids.len()],
        token_ids,
    )?;
    cases += 1;
    Ok(cases)
}

#[derive(Clone)]
struct RuntimeState {
    layout: StateLayout,
    weight: MetalTensor,
    output: MetalTensor,
}

#[derive(Debug, Serialize)]
struct SetupTiming {
    manifest_read_ms: f64,
    manifest_sha256_ms: f64,
    manifest_parse_ms: f64,
    manifest_validate_ms: f64,
    metadata_layout_ms: f64,
    row_copy_padding_source_hash_ms: f64,
    bank_sha256_ms: f64,
    metal_upload_ms: f64,
    weight_view_binding_ms: f64,
    compact_scratch_and_output_views_ms: f64,
    accounted_ms: f64,
    residual_ms: f64,
    total_b_ms: f64,
}

struct SetupArtifacts {
    manifest: ValidatedManifest,
    layout: BankLayout,
    bank: PreparedBank,
    bank_sha256: String,
    bank_buffer: Buffer,
    states: Vec<RuntimeState>,
    compact_storage: MetalTensor,
    timing: SetupTiming,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn setup_branch_bank(
    ctx: &MetalContext,
    manifest_path: &Path,
    source: &[u8],
    profile: ProfileSpec,
) -> Result<SetupArtifacts> {
    let total_start = Instant::now();

    let phase = Instant::now();
    let manifest_bytes = std::fs::read(manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;
    let manifest_read_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    ensure!(
        manifest_sha256 == MANIFEST_SHA256,
        "manifest SHA-256 mismatch: {manifest_sha256}"
    );
    let manifest_sha256_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let manifest_document: ManifestDocument =
        serde_json::from_slice(&manifest_bytes).context("parse frozen branch manifest")?;
    let manifest_parse_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let manifest = validate_manifest(manifest_document)?;
    let manifest_validate_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let layout = build_bank_layout(&manifest, profile)?;
    let metadata_layout_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let bank = prepare_bank(&layout, source, profile)?;
    let row_copy_padding_source_hash_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let bank_sha256 = sha256_hex(&bank.bytes);
    let bank_sha256_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let bank_buffer = ctx.buffer_from(&bank.bytes)?;
    let metal_upload_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let mut weight_views = Vec::with_capacity(layout.states.len());
    for state in &layout.states {
        let view = MetalTensor::q6_k_row_bank_weight_view(
            bank_buffer.clone(),
            u64::try_from(state.offset)?,
            profile.hidden,
            state.state.token_ids.len(),
        )?;
        ensure!(
            view.shape.as_slice() == [profile.hidden as u64, state.state.token_ids.len() as u64]
                && view.dtype == GgmlType::Q6_K
                && usize::try_from(view.offset)? == state.offset
                && usize::try_from(view.n_bytes())? == state.payload_bytes
                && !view.is_writable(),
            "Q6_K row-bank view contract mismatch"
        );
        weight_views.push(view);
    }
    let weight_view_binding_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let compact_storage = MetalTensor::zeros_f32(
        ctx,
        vec![(2 * OUTPUT_GUARD_ELEMENTS + WIDTHS[WIDTHS.len() - 1]) as u64],
    )?;
    let states = layout
        .states
        .iter()
        .cloned()
        .zip(weight_views)
        .map(|(layout, weight)| RuntimeState {
            output: compact_storage.view_subrange(
                OUTPUT_GUARD_ELEMENTS as u64,
                vec![layout.state.token_ids.len() as u64],
            ),
            layout,
            weight,
        })
        .collect();
    let compact_scratch_and_output_views_ms = elapsed_ms(phase);

    let total_b_ms = elapsed_ms(total_start);
    let accounted_ms = manifest_read_ms
        + manifest_sha256_ms
        + manifest_parse_ms
        + manifest_validate_ms
        + metadata_layout_ms
        + row_copy_padding_source_hash_ms
        + bank_sha256_ms
        + metal_upload_ms
        + weight_view_binding_ms
        + compact_scratch_and_output_views_ms;
    let residual_ms = total_b_ms - accounted_ms;
    Ok(SetupArtifacts {
        manifest,
        layout,
        bank,
        bank_sha256,
        bank_buffer,
        states,
        compact_storage,
        timing: SetupTiming {
            manifest_read_ms,
            manifest_sha256_ms,
            manifest_parse_ms,
            manifest_validate_ms,
            metadata_layout_ms,
            row_copy_padding_source_hash_ms,
            bank_sha256_ms,
            metal_upload_ms,
            weight_view_binding_ms,
            compact_scratch_and_output_views_ms,
            accounted_ms,
            residual_ms,
            total_b_ms,
        },
    })
}

#[derive(Debug, Serialize)]
struct CorrectnessSummary {
    hidden_seeds: Vec<u64>,
    full_dispatches: usize,
    compact_dispatches: usize,
    compared_logits: usize,
    sampler_cases: usize,
    source_rows: usize,
    source_rows_sha256: String,
    bank_sha256_before: String,
    bank_sha256_after: String,
    bit_exact: bool,
    guards_intact: bool,
    bank_unchanged: bool,
}

fn validate_gpu_correctness(
    ctx: &MetalContext,
    profile: ProfileSpec,
    source: &[u8],
    full_head: &MetalTensor,
    hidden_inputs: &[MetalTensor],
    full_output: &MetalTensor,
    setup: &SetupArtifacts,
) -> Result<CorrectnessSummary> {
    ensure!(hidden_inputs.len() == HIDDEN_SEEDS.len());
    let full_poison = vec![f32::from_bits(OUTPUT_POISON_BITS); VOCAB_ROWS];
    let mut compact_dispatches = 0usize;
    let mut compared_logits = 0usize;
    for (&seed, hidden) in HIDDEN_SEEDS.iter().zip(hidden_inputs) {
        write_f32_tensor(full_output, &full_poison)?;
        run_q6_dispatch(
            ctx,
            full_head,
            hidden,
            full_output,
            profile.hidden,
            VOCAB_ROWS,
            &format!("correctness full head seed {seed}"),
        )?;
        let full_logits = read_f32_tensor(full_output)?;
        ensure!(
            full_logits
                .iter()
                .all(|value| value.to_bits() != OUTPUT_POISON_BITS),
            "full head did not overwrite every logit"
        );
        for state in &setup.states {
            let width = state.layout.state.token_ids.len();
            write_compact_poison(&setup.compact_storage, width)?;
            run_q6_dispatch(
                ctx,
                &state.weight,
                hidden,
                &state.output,
                profile.hidden,
                width,
                &format!(
                    "correctness compact seed {seed} state {}",
                    state.layout.state.bank_state_index
                ),
            )?;
            let compact = read_f32_tensor(&state.output)?;
            ensure!(
                compact
                    .iter()
                    .all(|value| value.to_bits() != OUTPUT_POISON_BITS),
                "compact head did not overwrite every logit"
            );
            for (&token, &actual) in state.layout.state.token_ids.iter().zip(&compact) {
                let expected = full_logits[usize::try_from(token)?];
                ensure!(
                    actual.to_bits() == expected.to_bits(),
                    "compact/full mismatch at seed {seed}, state {}, token {token}: {actual:?} != {expected:?}",
                    state.layout.state.bank_state_index
                );
                compared_logits += 1;
            }
            validate_compact_guards(&setup.compact_storage, width)?;
            compact_dispatches += 1;
        }
    }

    let sampler_cases = validate_sampler_contract(&setup.manifest)?;
    let source_rows_sha256 = verify_source_rows(source, profile, &setup.bank.source_row_hashes)?;
    ensure!(
        source_rows_sha256 == setup.bank.source_rows_sha256,
        "source row set digest changed"
    );
    let uploaded = buffer_bytes(&setup.bank_buffer, setup.bank.bytes.len())?;
    ensure!(uploaded == setup.bank.bytes, "uploaded bank bytes changed");
    let bank_sha256_after = sha256_hex(&uploaded);
    ensure!(
        bank_sha256_after == setup.bank_sha256,
        "uploaded bank hash changed"
    );
    Ok(CorrectnessSummary {
        hidden_seeds: HIDDEN_SEEDS.to_vec(),
        full_dispatches: HIDDEN_SEEDS.len(),
        compact_dispatches,
        compared_logits,
        sampler_cases,
        source_rows: setup.bank.source_row_hashes.len(),
        source_rows_sha256,
        bank_sha256_before: setup.bank_sha256.clone(),
        bank_sha256_after,
        bit_exact: true,
        guards_intact: true,
        bank_unchanged: true,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Full,
    Compact,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "A_full_head",
            Self::Compact => "B_compact_bank",
        }
    }
}

#[derive(Debug, Serialize)]
struct ChargedTiming {
    command: &'static str,
    status: &'static str,
    error_none: bool,
    matvec_dispatches: usize,
    lookup_ms: f64,
    create_encode_ms: f64,
    commit_wait_status_ms: f64,
    gpu_ms: f64,
    readback_gather_ms: f64,
    sampler_mapping_ms: f64,
    unattributed_ms: f64,
    total_wall_ms: f64,
}

#[derive(Debug, Serialize)]
struct TimedArmObservation {
    arm: &'static str,
    predecessor: &'static str,
    conditioning: DispatchTiming,
    timing: ChargedTiming,
    selection: MappedOutcome,
}

#[derive(Debug, Serialize)]
struct PairObservation {
    phase: &'static str,
    round: usize,
    width_order_position: usize,
    width: usize,
    bank_state_index: usize,
    pair_id: usize,
    direction: &'static str,
    first: TimedArmObservation,
    second: TimedArmObservation,
}

#[derive(Debug, Serialize)]
struct ScreenObservation {
    round: usize,
    width_order_position: usize,
    width: usize,
    bank_state_index: usize,
    observation: TimedArmObservation,
}

fn gather_f32(tensor: &MetalTensor, token_ids: &[i32]) -> Result<Vec<f32>> {
    ensure!(tensor.dtype == GgmlType::F32);
    let count = usize::try_from(tensor.n_elements())?;
    let (start, _) = tensor_byte_range(tensor)?;
    let contents = tensor.buffer.contents();
    let values = unsafe {
        std::slice::from_raw_parts(
            contents.as_ptr().cast::<u8>().add(start).cast::<f32>(),
            count,
        )
    };
    token_ids
        .iter()
        .map(|&token| {
            let index = usize::try_from(token).context("negative gather token")?;
            values
                .get(index)
                .copied()
                .context("gather token is out of range")
        })
        .collect()
}

fn run_timed_arm(
    ctx: &MetalContext,
    arm: Arm,
    conditioning: DispatchTiming,
    profile: ProfileSpec,
    states: &[RuntimeState],
    bank_state_index: usize,
    full_head: &MetalTensor,
    hidden: &MetalTensor,
    full_output: &MetalTensor,
) -> Result<TimedArmObservation> {
    let mut sampler = Sampler::new(SamplingConfig::qwen_chat(0))?;
    let total_start = Instant::now();

    let phase = Instant::now();
    let state = states
        .get(bank_state_index)
        .context("timed bank state index is out of range")?;
    let width = state.layout.state.token_ids.len();
    let (weight, output, n_out) = match arm {
        Arm::Full => (full_head, full_output, VOCAB_ROWS),
        Arm::Compact => (&state.weight, &state.output, width),
    };
    let lookup_ms = elapsed_ms(phase);

    validate_dispatch(weight, hidden, output, profile.hidden, n_out)?;
    let phase = Instant::now();
    let cmd = ctx
        .queue
        .commandBuffer()
        .with_context(|| format!("{} timed command buffer", arm.label()))?;
    let enc = KernelEncoder::begin(&cmd);
    encode_mat_vec_q6_k_f32(ctx, &enc, weight, hidden, output, profile.hidden, n_out)?;
    enc.end();
    let create_encode_ms = elapsed_ms(phase);

    let phase = Instant::now();
    cmd.commit();
    cmd.waitUntilCompleted();
    check_command(&cmd, arm.label())?;
    let gpu_ms = command_gpu_ms(&cmd)?;
    let commit_wait_status_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let logits = match arm {
        Arm::Full => gather_f32(full_output, &state.layout.state.token_ids)?,
        Arm::Compact => read_f32_tensor(&state.output)?,
    };
    let readback_gather_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let selection =
        sample_compact_with_sampler(&mut sampler, &logits, &state.layout.state.token_ids)?;
    ensure!(
        matches!(selection, MappedOutcome::Token { .. }),
        "timed sampler did not select a token: {selection:?}"
    );
    let sampler_mapping_ms = elapsed_ms(phase);
    let total_wall_ms = elapsed_ms(total_start);
    let accounted = lookup_ms
        + create_encode_ms
        + commit_wait_status_ms
        + readback_gather_ms
        + sampler_mapping_ms;
    Ok(TimedArmObservation {
        arm: arm.label(),
        predecessor: "full_head_conditioning",
        conditioning,
        timing: ChargedTiming {
            command: "kernel_mat_vec_q6_K_f32",
            status: "completed",
            error_none: true,
            matvec_dispatches: 1,
            lookup_ms,
            create_encode_ms,
            commit_wait_status_ms,
            gpu_ms,
            readback_gather_ms,
            sampler_mapping_ms,
            unattributed_ms: (total_wall_ms - accounted).max(0.0),
            total_wall_ms,
        },
        selection,
    })
}

fn conditioning_dispatch(
    ctx: &MetalContext,
    profile: ProfileSpec,
    full_head: &MetalTensor,
    hidden: &MetalTensor,
    full_output: &MetalTensor,
) -> Result<DispatchTiming> {
    run_q6_dispatch(
        ctx,
        full_head,
        hidden,
        full_output,
        profile.hidden,
        VOCAB_ROWS,
        "full-head conditioning",
    )
}

fn ordered_widths(round: usize) -> [usize; WIDTHS.len()] {
    let mut order = WIDTHS;
    order.rotate_left(round % WIDTHS.len());
    order
}

fn arm_order(round: usize) -> [Arm; 2] {
    if round.is_multiple_of(2) {
        [Arm::Full, Arm::Compact]
    } else {
        [Arm::Compact, Arm::Full]
    }
}

fn observation_for_arm(pair: &PairObservation, arm: Arm) -> &TimedArmObservation {
    if pair.first.arm == arm.label() {
        &pair.first
    } else {
        &pair.second
    }
}

fn median(values: &[f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "median requires observations");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "median received nonfinite value"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn weighted_ms(
    per_width: &BTreeMap<usize, f64>,
    histogram: &BTreeMap<usize, usize>,
) -> Result<f64> {
    let total: usize = histogram.values().sum();
    ensure!(total > 0, "weighted total is empty");
    let mut sum = 0.0;
    for (&width, &count) in histogram {
        sum += per_width
            .get(&width)
            .with_context(|| format!("missing width {width} median"))?
            * count as f64;
    }
    Ok(sum / total as f64)
}

#[derive(Clone, Copy, Debug)]
struct FrozenHeadGateDecision {
    head_removal: f64,
    whole_token_saving: f64,
    head_removal_pass: bool,
    whole_token_saving_pass: bool,
    equivalent_limit_ms: f64,
    equivalent_diagnostic: bool,
}

fn frozen_head_gate_decision(h0_ms: f64, h1_ms: f64, t0_ms: f64) -> FrozenHeadGateDecision {
    let head_removal = 1.0 - h1_ms / h0_ms;
    let whole_token_saving = (h0_ms - h1_ms) / t0_ms;
    let head_removal_pass = head_removal >= 0.70;
    let whole_token_saving_pass = whole_token_saving >= 0.05;
    let equivalent_limit_ms = (0.30 * h0_ms).min(h0_ms - 0.05 * t0_ms);
    FrozenHeadGateDecision {
        head_removal,
        whole_token_saving,
        head_removal_pass,
        whole_token_saving_pass,
        equivalent_limit_ms,
        equivalent_diagnostic: h1_ms <= equivalent_limit_ms,
    }
}

fn run_pair(
    phase: &'static str,
    round: usize,
    width_order_position: usize,
    width: usize,
    pair_id: usize,
    ctx: &MetalContext,
    profile: ProfileSpec,
    setup: &SetupArtifacts,
    full_head: &MetalTensor,
    hidden: &MetalTensor,
    full_output: &MetalTensor,
) -> Result<PairObservation> {
    let bank_state_index = *setup
        .manifest
        .representative_by_width
        .get(&width)
        .with_context(|| format!("missing representative for width {width}"))?;
    let order = arm_order(round);
    let first_conditioning = conditioning_dispatch(ctx, profile, full_head, hidden, full_output)?;
    let first = run_timed_arm(
        ctx,
        order[0],
        first_conditioning,
        profile,
        &setup.states,
        bank_state_index,
        full_head,
        hidden,
        full_output,
    )?;
    let second_conditioning = conditioning_dispatch(ctx, profile, full_head, hidden, full_output)?;
    let second = run_timed_arm(
        ctx,
        order[1],
        second_conditioning,
        profile,
        &setup.states,
        bank_state_index,
        full_head,
        hidden,
        full_output,
    )?;
    ensure!(
        first.selection == second.selection,
        "paired arms selected different tokens at width {width}"
    );
    Ok(PairObservation {
        phase,
        round: round + 1,
        width_order_position: width_order_position + 1,
        width,
        bank_state_index,
        pair_id,
        direction: if order[0] == Arm::Full { "AB" } else { "BA" },
        first,
        second,
    })
}

#[derive(Debug, Serialize)]
struct WidthSummary {
    width: usize,
    canonical_incidences: usize,
    state_count: usize,
    trace_incidences: usize,
    a_median_wall_ms: f64,
    b_median_wall_ms: f64,
    median_paired_saving_ms: f64,
    a_median_gpu_ms: f64,
    b_median_gpu_ms: f64,
}

#[derive(Debug, Serialize)]
struct PathNet {
    path_index: usize,
    transition_saving_ms: f64,
    setup_b_ms: f64,
    net_ms: f64,
}

struct PairAnalysis {
    summaries: Vec<WidthSummary>,
    a_by_width: BTreeMap<usize, f64>,
    b_by_width: BTreeMap<usize, f64>,
    canonical_h0_ms: f64,
    canonical_h1_ms: f64,
    uniform_state_h0_ms: f64,
    uniform_state_h1_ms: f64,
    trace_h0_ms: f64,
    trace_h1_ms: f64,
    trace_strata: BTreeMap<String, (f64, f64)>,
    path_nets: Vec<PathNet>,
    minimum_path_net_ms: f64,
    uniform_path_mean_net_ms: f64,
    all_widths_positive: bool,
}

fn analyze_pairs(pairs: &[PairObservation], setup: &SetupArtifacts) -> Result<PairAnalysis> {
    ensure!(
        pairs.len() == 12 * WIDTHS.len(),
        "paired sample count mismatch"
    );
    let mut summaries = Vec::with_capacity(WIDTHS.len());
    let mut a_by_width = BTreeMap::new();
    let mut b_by_width = BTreeMap::new();
    let mut all_widths_positive = true;
    for width in WIDTHS {
        let selected: Vec<_> = pairs.iter().filter(|pair| pair.width == width).collect();
        ensure!(selected.len() == 12, "paired width sample count mismatch");
        let a_wall: Vec<_> = selected
            .iter()
            .map(|pair| observation_for_arm(pair, Arm::Full).timing.total_wall_ms)
            .collect();
        let b_wall: Vec<_> = selected
            .iter()
            .map(|pair| observation_for_arm(pair, Arm::Compact).timing.total_wall_ms)
            .collect();
        let paired_savings: Vec<_> = selected
            .iter()
            .map(|pair| {
                observation_for_arm(pair, Arm::Full).timing.total_wall_ms
                    - observation_for_arm(pair, Arm::Compact).timing.total_wall_ms
            })
            .collect();
        let a_gpu: Vec<_> = selected
            .iter()
            .map(|pair| observation_for_arm(pair, Arm::Full).timing.gpu_ms)
            .collect();
        let b_gpu: Vec<_> = selected
            .iter()
            .map(|pair| observation_for_arm(pair, Arm::Compact).timing.gpu_ms)
            .collect();
        let a_median = median(&a_wall)?;
        let b_median = median(&b_wall)?;
        let paired_median = median(&paired_savings)?;
        all_widths_positive &= paired_median > 0.0;
        a_by_width.insert(width, a_median);
        b_by_width.insert(width, b_median);
        summaries.push(WidthSummary {
            width,
            canonical_incidences: setup
                .manifest
                .canonical_histogram
                .get(&width)
                .copied()
                .unwrap_or(0),
            state_count: setup
                .manifest
                .row_histogram
                .get(&width)
                .copied()
                .unwrap_or(0),
            trace_incidences: setup
                .manifest
                .trace_histogram
                .get(&width)
                .copied()
                .unwrap_or(0),
            a_median_wall_ms: a_median,
            b_median_wall_ms: b_median,
            median_paired_saving_ms: paired_median,
            a_median_gpu_ms: median(&a_gpu)?,
            b_median_gpu_ms: median(&b_gpu)?,
        });
    }

    let canonical_h0_ms = weighted_ms(&a_by_width, &setup.manifest.canonical_histogram)?;
    let canonical_h1_ms = weighted_ms(&b_by_width, &setup.manifest.canonical_histogram)?;
    let uniform_state_h0_ms = weighted_ms(&a_by_width, &setup.manifest.row_histogram)?;
    let uniform_state_h1_ms = weighted_ms(&b_by_width, &setup.manifest.row_histogram)?;
    let trace_h0_ms = weighted_ms(&a_by_width, &setup.manifest.trace_histogram)?;
    let trace_h1_ms = weighted_ms(&b_by_width, &setup.manifest.trace_histogram)?;
    let trace_strata = setup
        .manifest
        .trace_stratum_histograms
        .iter()
        .map(|(stratum, histogram)| {
            Ok((
                stratum.clone(),
                (
                    weighted_ms(&a_by_width, histogram)?,
                    weighted_ms(&b_by_width, histogram)?,
                ),
            ))
        })
        .collect::<Result<_>>()?;

    let mut path_nets = Vec::with_capacity(setup.manifest.canonical_paths.len());
    for path in &setup.manifest.canonical_paths {
        let transition_saving_ms = path.widths.iter().try_fold(0.0, |sum, width| {
            Ok::<_, anyhow::Error>(
                sum + a_by_width
                    .get(width)
                    .with_context(|| format!("missing A median for width {width}"))?
                    - b_by_width
                        .get(width)
                        .with_context(|| format!("missing B median for width {width}"))?,
            )
        })?;
        path_nets.push(PathNet {
            path_index: path.path_index,
            transition_saving_ms,
            setup_b_ms: setup.timing.total_b_ms,
            net_ms: transition_saving_ms - setup.timing.total_b_ms,
        });
    }
    let minimum_path_net_ms = path_nets
        .iter()
        .map(|path| path.net_ms)
        .min_by(f64::total_cmp)
        .context("no canonical path nets")?;
    let uniform_path_mean_net_ms =
        path_nets.iter().map(|path| path.net_ms).sum::<f64>() / path_nets.len() as f64;
    Ok(PairAnalysis {
        summaries,
        a_by_width,
        b_by_width,
        canonical_h0_ms,
        canonical_h1_ms,
        uniform_state_h0_ms,
        uniform_state_h1_ms,
        trace_h0_ms,
        trace_h1_ms,
        trace_strata,
        path_nets,
        minimum_path_net_ms,
        uniform_path_mean_net_ms,
        all_widths_positive,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_dense_prerequisite(path: &Path, build_identity: &Value) -> Result<Value> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read prior A3B result {}", path.display()))?;
    let result: Value = serde_json::from_slice(&bytes).context("parse prior A3B result")?;
    ensure!(
        result.get("schema").and_then(Value::as_str) == Some(RESULT_SCHEMA),
        "prior A3B result schema mismatch"
    );
    ensure!(
        result.pointer("/profile/id").and_then(Value::as_str) == Some("a3b"),
        "prior result is not A3B"
    );
    ensure!(
        result.get("disposition").and_then(Value::as_str) == Some("go"),
        "prior A3B result did not clear GO"
    );
    ensure!(
        result.get("authority").and_then(Value::as_str) == Some("one_later_a3b_integrated_packet")
            && result
                .pointer("/gates/all_go_gates")
                .and_then(Value::as_bool)
                == Some(true)
            && result
                .pointer("/correctness/bit_exact")
                .and_then(Value::as_bool)
                == Some(true)
            && result
                .pointer("/post_timing_validation/bank_unchanged")
                .and_then(Value::as_bool)
                == Some(true),
        "prior A3B authority or validity summary mismatch"
    );
    ensure!(
        result
            .pointer("/model_identity/sha256")
            .and_then(Value::as_str)
            == Some(GrammarLmHeadProfile::A3b.spec().model_sha256),
        "prior A3B model identity mismatch"
    );
    ensure!(
        result
            .pointer("/manifest_identity/sha256")
            .and_then(Value::as_str)
            == Some(MANIFEST_SHA256),
        "prior A3B result used a different manifest"
    );
    ensure!(
        result.get("build_identity") == Some(build_identity),
        "prior A3B result build identity differs"
    );
    ensure!(
        result
            .pointer("/protocol/warmup_rounds")
            .and_then(Value::as_u64)
            == Some(5)
            && result
                .pointer("/protocol/screen_rounds")
                .and_then(Value::as_u64)
                == Some(12)
            && result
                .pointer("/protocol/paired_rounds")
                .and_then(Value::as_u64)
                == Some(12),
        "prior A3B result protocol mismatch"
    );
    Ok(json!({
        "path": path,
        "sha256": sha256_hex(&bytes),
        "validated_profile": "a3b",
        "validated_disposition": "go",
    }))
}

fn profile_json(profile: ProfileSpec) -> Value {
    json!({
        "id": profile.id,
        "expected_path": profile.expected_path,
        "file_bytes": profile.file_bytes,
        "model_sha256": profile.model_sha256,
        "architecture": match profile.arch_kind {
            ArchKind::Dense => "dense",
            ArchKind::Moe => "moe",
        },
        "layers": profile.layers,
        "hidden": profile.hidden,
        "vocab": VOCAB_ROWS,
        "head_bytes": profile.head_bytes,
        "row_bytes": profile.row_bytes,
        "bank_payload_bytes": profile.bank_payload_bytes,
        "bank_padding_bytes": profile.bank_padding_bytes,
        "bank_span_bytes": profile.bank_span_bytes,
        "t0_ms": profile.t0_ms,
    })
}

pub fn run(
    args: GrammarLmHeadRowFloorArgs,
    build_identity: Value,
    qwen_env: BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        args.warmup_rounds == 5 && args.screen_rounds == 12 && args.paired_rounds == 12,
        "canonical floor requires warmup=5, screen=12, paired=12"
    );
    let profile = args.profile.spec();
    let dense_prerequisite = match (args.profile, args.a3b_result.as_deref()) {
        (GrammarLmHeadProfile::A3b, None) => Value::Null,
        (GrammarLmHeadProfile::A3b, Some(_)) => {
            bail!("--a3b-result is forbidden for the A3B primary run")
        }
        (GrammarLmHeadProfile::Dense, Some(path)) => {
            validate_dense_prerequisite(path, &build_identity)?
        }
        (GrammarLmHeadProfile::Dense, None) => {
            bail!("dense guard requires --a3b-result from the same build")
        }
    };

    let canonical_model = std::fs::canonicalize(&args.model)
        .with_context(|| format!("canonicalize {}", args.model.display()))?;
    let canonical_expected = std::fs::canonicalize(profile.expected_path)
        .with_context(|| format!("canonicalize {}", profile.expected_path))?;
    ensure!(
        canonical_model == canonical_expected,
        "model path does not match the frozen profile"
    );
    let metadata = std::fs::metadata(&canonical_model)
        .with_context(|| format!("stat {}", canonical_model.display()))?;
    ensure!(
        metadata.len() == profile.file_bytes,
        "model file size mismatch"
    );

    let gguf_open_start = Instant::now();
    let gguf = GgufFile::open(&canonical_model)?;
    let model = Model::from_gguf(&gguf)?;
    let gguf_open_ms = elapsed_ms(gguf_open_start);
    ensure!(
        gguf.shard_count() == 1,
        "v0.655 requires a single-shard GGUF"
    );

    let model_hash_start = Instant::now();
    let model_sha256 = sha256_file(&canonical_model)?;
    let model_hash_ms = elapsed_ms(model_hash_start);
    ensure!(
        model_sha256 == profile.model_sha256,
        "model SHA-256 mismatch"
    );
    ensure!(
        model.arch.kind == profile.arch_kind
            && model.arch.n_layer == profile.layers
            && model.arch.hidden_size as usize == profile.hidden
            && model.arch.vocab_size as usize == VOCAB_ROWS,
        "model architecture does not match the frozen profile"
    );
    ensure!(!model.tied_embeddings, "v0.655 requires an untied lm_head");
    let descriptor = model.lm_head;
    ensure!(
        descriptor.name == "output.weight"
            && descriptor.dtype == GgmlType::Q6_K
            && descriptor.shape.as_slice() == [profile.hidden as u64, VOCAB_ROWS as u64]
            && descriptor.n_bytes as usize == profile.head_bytes,
        "output.weight descriptor does not match the frozen profile"
    );
    ensure!(
        profile.hidden / Q6_BLOCK_ELEMENTS * Q6_BLOCK_BYTES == profile.row_bytes
            && profile.row_bytes * VOCAB_ROWS == profile.head_bytes,
        "frozen profile Q6_K geometry is inconsistent"
    );

    let source = gguf.try_slice(descriptor)?;
    ensure!(source.len() == profile.head_bytes);
    let tensor_hash_start = Instant::now();
    let tensor_sha256 = sha256_hex(source);
    let tensor_hash_ms = elapsed_ms(tensor_hash_start);

    let metal_start = Instant::now();
    let ctx = MetalContext::new()?;
    let metal_init_ms = elapsed_ms(metal_start);
    let full_upload_start = Instant::now();
    let full_head = MetalTensor::from_bytes(
        &ctx,
        source,
        vec![profile.hidden as u64, VOCAB_ROWS as u64],
        GgmlType::Q6_K,
    )?;
    let full_head_upload_ms = elapsed_ms(full_upload_start);

    let fixture_start = Instant::now();
    let mut hidden_inputs = Vec::with_capacity(HIDDEN_SEEDS.len());
    for seed in HIDDEN_SEEDS {
        let hidden = hidden_vector(seed, profile.hidden);
        ensure!(hidden.iter().any(|&value| value != 0.0));
        hidden_inputs.push(MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&hidden),
            vec![profile.hidden as u64],
            GgmlType::F32,
        )?);
    }
    let full_output = MetalTensor::zeros_f32(&ctx, vec![VOCAB_ROWS as u64])?;
    let fixture_setup_ms = elapsed_ms(fixture_start);

    let pipeline_start = Instant::now();
    let pipeline = ctx.pipeline("kernel_mat_vec_q6_K_f32")?;
    let pipeline_compile_ms = elapsed_ms(pipeline_start);
    let pipeline_info = json!({
        "kernel": "kernel_mat_vec_q6_K_f32",
        "thread_execution_width": pipeline.threadExecutionWidth(),
        "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup(),
        "static_threadgroup_memory_length": pipeline.staticThreadgroupMemoryLength(),
    });
    drop(pipeline);

    let setup = setup_branch_bank(&ctx, &args.manifest, source, profile)?;
    let correctness_start = Instant::now();
    let correctness = validate_gpu_correctness(
        &ctx,
        profile,
        source,
        &full_head,
        &hidden_inputs,
        &full_output,
        &setup,
    )?;
    let correctness_ms = elapsed_ms(correctness_start);
    let timed_hidden = &hidden_inputs[0];

    let mut warmup_pairs = Vec::with_capacity(args.warmup_rounds * WIDTHS.len());
    let mut pair_id = 0usize;
    for round in 0..args.warmup_rounds {
        for (position, width) in ordered_widths(round).into_iter().enumerate() {
            pair_id += 1;
            warmup_pairs.push(run_pair(
                "warmup",
                round,
                position,
                width,
                pair_id,
                &ctx,
                profile,
                &setup,
                &full_head,
                timed_hidden,
                &full_output,
            )?);
        }
    }

    let mut screen_samples = Vec::with_capacity(args.screen_rounds * WIDTHS.len());
    for round in 0..args.screen_rounds {
        for (position, width) in ordered_widths(round).into_iter().enumerate() {
            let bank_state_index = setup.manifest.representative_by_width[&width];
            let conditioning =
                conditioning_dispatch(&ctx, profile, &full_head, timed_hidden, &full_output)?;
            let observation = run_timed_arm(
                &ctx,
                Arm::Full,
                conditioning,
                profile,
                &setup.states,
                bank_state_index,
                &full_head,
                timed_hidden,
                &full_output,
            )?;
            screen_samples.push(ScreenObservation {
                round: round + 1,
                width_order_position: position + 1,
                width,
                bank_state_index,
                observation,
            });
        }
    }
    let mut screen_medians = BTreeMap::new();
    for width in WIDTHS {
        let walls: Vec<_> = screen_samples
            .iter()
            .filter(|sample| sample.width == width)
            .map(|sample| sample.observation.timing.total_wall_ms)
            .collect();
        ensure!(walls.len() == args.screen_rounds);
        screen_medians.insert(width, median(&walls)?);
    }
    let screen_h0_ms = weighted_ms(&screen_medians, &setup.manifest.canonical_histogram)?;
    let screen_fraction = screen_h0_ms / profile.t0_ms;

    let mut paired_samples = Vec::new();
    if screen_fraction > 0.05 {
        paired_samples.reserve(args.paired_rounds * WIDTHS.len());
        for round in 0..args.paired_rounds {
            for (position, width) in ordered_widths(round).into_iter().enumerate() {
                pair_id += 1;
                paired_samples.push(run_pair(
                    "paired",
                    round,
                    position,
                    width,
                    pair_id,
                    &ctx,
                    profile,
                    &setup,
                    &full_head,
                    timed_hidden,
                    &full_output,
                )?);
            }
        }
    }

    validate_compact_outer_guards(&setup.compact_storage)?;
    let final_bank = buffer_bytes(&setup.bank_buffer, setup.bank.bytes.len())?;
    ensure!(final_bank == setup.bank.bytes, "timing mutated the bank");
    let final_bank_sha256 = sha256_hex(&final_bank);
    ensure!(final_bank_sha256 == setup.bank_sha256);

    let analysis = if paired_samples.is_empty() {
        None
    } else {
        Some(analyze_pairs(&paired_samples, &setup)?)
    };
    let (disposition, authority, gates, economics, paired_summary) = match &analysis {
        None => (
            "screen_stop",
            "none",
            json!({
                "screen_h0_over_t0_gt_5pct": false,
                "all_other_go_gates": null,
            }),
            Value::Null,
            Value::Null,
        ),
        Some(analysis) => {
            let decision = frozen_head_gate_decision(
                analysis.canonical_h0_ms,
                analysis.canonical_h1_ms,
                profile.t0_ms,
            );
            let all_paths_positive = analysis.minimum_path_net_ms > 0.0;
            let go = analysis.all_widths_positive
                && decision.head_removal_pass
                && decision.whole_token_saving_pass
                && all_paths_positive;
            let trace_strata: BTreeMap<_, _> = analysis
                .trace_strata
                .iter()
                .map(|(stratum, &(h0, h1))| {
                    (
                        stratum.clone(),
                        json!({"h0_ms": h0, "h1_ms": h1, "saving_ms": h0 - h1}),
                    )
                })
                .collect();
            (
                if go { "go" } else { "no_go" },
                if go {
                    if profile.id == "a3b" {
                        "one_later_a3b_integrated_packet"
                    } else {
                        "dense_transfer_guard_cleared"
                    }
                } else {
                    "none"
                },
                json!({
                    "screen_h0_over_t0_gt_5pct": true,
                    "positive_paired_median_saving_all_widths": analysis.all_widths_positive,
                    "head_removal_gte_70pct": decision.head_removal_pass,
                    "whole_token_saving_gte_5pct": decision.whole_token_saving_pass,
                    "all_36_path_nets_positive": all_paths_positive,
                    "equivalent_h1_limit_ms_diagnostic": decision.equivalent_limit_ms,
                    "equivalent_h1_diagnostic": decision.equivalent_diagnostic,
                    "equivalent_diagnostic_is_not_a_gate": true,
                    "all_go_gates": go,
                }),
                json!({
                    "canonical": {
                        "incidences": 648,
                        "h0_ms": analysis.canonical_h0_ms,
                        "h1_ms": analysis.canonical_h1_ms,
                        "saving_ms": analysis.canonical_h0_ms - analysis.canonical_h1_ms,
                        "head_removal": decision.head_removal,
                        "whole_token_saving": decision.whole_token_saving,
                        "h0_over_t0": analysis.canonical_h0_ms / profile.t0_ms,
                    },
                    "uniform_states": {
                        "states": 451,
                        "h0_ms": analysis.uniform_state_h0_ms,
                        "h1_ms": analysis.uniform_state_h1_ms,
                    },
                    "trace": {
                        "incidences": 360,
                        "h0_ms": analysis.trace_h0_ms,
                        "h1_ms": analysis.trace_h1_ms,
                        "strata": trace_strata,
                    },
                    "path_nets": analysis.path_nets,
                    "minimum_path_net_ms": analysis.minimum_path_net_ms,
                    "uniform_path_mean_net_ms": analysis.uniform_path_mean_net_ms,
                    "setup_b_ms": setup.timing.total_b_ms,
                }),
                json!({
                    "per_width": analysis.summaries,
                    "a_medians_by_width_ms": analysis.a_by_width,
                    "b_medians_by_width_ms": analysis.b_by_width,
                    "pairs": paired_samples,
                }),
            )
        }
    };

    let result = json!({
        "schema": RESULT_SCHEMA,
        "test": "grammar_lm_head_row_floor",
        "claim_scope": "charged exact primitive floor; no grammar runtime or product speedup authority",
        "test_time": super::utc_iso8601_now(),
        "build_identity": build_identity,
        "qwen_env": qwen_env,
        "profile": profile_json(profile),
        "dense_prerequisite": dense_prerequisite,
        "model_identity": {
            "path": canonical_model,
            "file_bytes": metadata.len(),
            "sha256": model_sha256,
            "sha256_ms": model_hash_ms,
            "gguf_open_and_bind_ms": gguf_open_ms,
            "shards": gguf.shard_count(),
            "total_mapped_bytes": gguf.total_mapped_len(),
        },
        "tensor_identity": {
            "name": descriptor.name,
            "dtype": "Q6_K",
            "shape": descriptor.shape,
            "bytes": source.len(),
            "sha256": tensor_sha256,
            "sha256_ms": tensor_hash_ms,
        },
        "manifest_identity": {
            "path": args.manifest,
            "sha256": MANIFEST_SHA256,
            "schema": MANIFEST_SCHEMA,
            "fingerprints": setup.manifest.fingerprints,
            "counts": setup.manifest.counts,
            "trace_source": setup.manifest.trace_source,
        },
        "common_fixture_costs": {
            "metal_init_ms": metal_init_ms,
            "full_head_upload_ms": full_head_upload_ms,
            "hidden_and_full_output_setup_ms": fixture_setup_ms,
            "pipeline_compile_ms": pipeline_compile_ms,
            "correctness_ms": correctness_ms,
            "excluded_from_b": true,
        },
        "pipeline": pipeline_info,
        "device": ctx.describe(),
        "bank": {
            "alignment_bytes": BANK_ALIGNMENT,
            "poison_byte": BANK_POISON,
            "states": setup.layout.states.len(),
            "row_incidences": setup.manifest.counts.branch_state_token_incidences,
            "unique_source_rows": setup.bank.source_row_hashes.len(),
            "row_bytes": profile.row_bytes,
            "payload_bytes": setup.layout.payload_bytes,
            "padding_bytes": setup.layout.padding_bytes,
            "span_bytes": setup.layout.span_bytes,
            "sha256": setup.bank_sha256,
            "source_rows_sha256": setup.bank.source_rows_sha256,
            "trailing_padding_included": true,
        },
        "incremental_setup": setup.timing,
        "correctness": correctness,
        "post_timing_validation": {
            "outer_guards_intact": true,
            "bank_unchanged": true,
            "bank_sha256": final_bank_sha256,
        },
        "protocol": {
            "sampler_version": SAMPLER_ALGORITHM_VERSION,
            "timed_sampler": "qwen_chat(0) at draw zero",
            "widths": WIDTHS,
            "representative_bank_state_by_width": setup.manifest.representative_by_width,
            "warmup_rounds": args.warmup_rounds,
            "screen_rounds": args.screen_rounds,
            "paired_rounds": args.paired_rounds,
            "width_order": "left rotation by zero-based round",
            "pair_order": "AB on even zero-based rounds, BA on odd rounds",
            "conditioning": "one separate completed full-head dispatch before every arm",
            "timed_command_buffer_dispatches": 1,
            "primary_clock": "wall before checked state lookup through sampler mapping",
            "gpu_clock": "MTLCommandBuffer GPUStartTime/GPUEndTime diagnostic",
        },
        "warmup": {
            "scored": false,
            "pairs": warmup_pairs,
        },
        "screen": {
            "h0_medians_by_width_ms": screen_medians,
            "weighted_h0_ms": screen_h0_ms,
            "h0_over_t0": screen_fraction,
            "continue_rule": "strictly greater than 0.05",
            "continued": screen_fraction > 0.05,
            "samples": screen_samples,
        },
        "paired": paired_summary,
        "economics": economics,
        "gates": gates,
        "disposition": disposition,
        "authority": authority,
    });
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROZEN_MANIFEST: &[u8] =
        include_bytes!("../../../docs/bench/v0655-grammar-branch-manifest.json");

    fn parse_frozen() -> ManifestDocument {
        serde_json::from_slice(FROZEN_MANIFEST).expect("parse frozen manifest")
    }

    fn mutate_manifest(mutator: impl FnOnce(&mut Value)) -> Result<ValidatedManifest> {
        let mut value: Value = serde_json::from_slice(FROZEN_MANIFEST)?;
        mutator(&mut value);
        let document: ManifestDocument = serde_json::from_value(value)?;
        validate_manifest(document)
    }

    #[test]
    fn frozen_manifest_reproduces_both_bank_layouts() {
        assert_eq!(sha256_hex(FROZEN_MANIFEST), MANIFEST_SHA256);
        let manifest = validate_manifest(parse_frozen()).expect("validate frozen manifest");
        assert_eq!(manifest.states.len(), 451);
        assert_eq!(manifest.representative_by_width.len(), WIDTHS.len());
        assert_eq!(manifest.canonical_histogram, expected_canonical_histogram());
        assert_eq!(manifest.trace_histogram, expected_trace_histogram());
        for profile in [
            GrammarLmHeadProfile::A3b.spec(),
            GrammarLmHeadProfile::Dense.spec(),
        ] {
            let layout = build_bank_layout(&manifest, profile).expect("build frozen layout");
            assert_eq!(layout.payload_bytes, profile.bank_payload_bytes);
            assert_eq!(layout.padding_bytes, profile.bank_padding_bytes);
            assert_eq!(layout.span_bytes, profile.bank_span_bytes);
            assert_eq!(layout.states.last().unwrap().offset % BANK_ALIGNMENT, 0);
            assert!(
                layout.states.last().unwrap().span_bytes
                    >= layout.states.last().unwrap().payload_bytes
            );
        }
    }

    #[test]
    fn manifest_rejects_index_order_and_duplicate_rows() {
        assert!(
            mutate_manifest(|value| value["states"][1]["bank_state_index"] = json!(9)).is_err()
        );
        assert!(
            mutate_manifest(|value| {
                value["states"][1]["topology_state_index"] =
                    value["states"][0]["topology_state_index"].clone()
            })
            .is_err()
        );
        assert!(
            mutate_manifest(|value| {
                let first = value["states"][0]["token_ids"][0].clone();
                value["states"][0]["token_ids"][1] = first;
            })
            .is_err()
        );
    }

    #[test]
    fn manifest_rejects_bad_representatives_and_path_mapping() {
        assert!(
            mutate_manifest(|value| {
                value["representative_bank_state_by_width"]["2"] = json!(1)
            })
            .is_err()
        );
        assert!(
            mutate_manifest(|value| {
                value["canonical_paths"][0]["bank_state_indices"][0] = json!(1)
            })
            .is_err()
        );
        assert!(
            mutate_manifest(|value| value["trace_records"][0]["canonical_path_index"] = json!(99))
                .is_err()
        );
    }

    #[test]
    fn manifest_rejects_prefix_and_digest_corruption() {
        assert!(mutate_manifest(|value| value["states"][0]["prefix_hex"] = json!("00")).is_err());
        assert!(
            mutate_manifest(|value| {
                value["fingerprints"]["unique_branch_token_rows_sha256"] = json!("00".repeat(32))
            })
            .is_err()
        );
        let mut raw = FROZEN_MANIFEST.to_vec();
        raw[0] ^= 1;
        assert_ne!(sha256_hex(&raw), MANIFEST_SHA256);
    }

    #[test]
    fn compact_sampler_matches_full_mask_for_frozen_cases() {
        let manifest = validate_manifest(parse_frozen()).expect("validate frozen manifest");
        assert_eq!(validate_sampler_contract(&manifest).unwrap(), 27);
    }

    #[test]
    fn all_negative_infinity_is_rejected_before_sampler_draw() {
        let ids = [90, 4754];
        let logits = [f32::NEG_INFINITY; 2];
        let mut sampler = Sampler::new(SamplingConfig::qwen_chat(42)).unwrap();
        let outcome = sample_compact_with_sampler(&mut sampler, &logits, &ids).unwrap();
        assert_eq!(
            outcome,
            MappedOutcome::Error {
                error: "all_negative_infinity".to_string(),
                original_token: None,
                draws_before: 0,
                draws_after: 0,
            }
        );
        assert_eq!(sampler.draws(), 0);
    }

    #[test]
    fn hidden_vectors_follow_frozen_wrapping_recurrence() {
        let values = hidden_vector(1, 4);
        assert_eq!(values.len(), 4);
        assert!(values.iter().any(|&value| value != 0.0));
        assert_eq!(values, hidden_vector(1, 4));
        assert_ne!(values, hidden_vector(2, 4));
        assert!(values.iter().all(|value| (-1.0..1.0).contains(value)));
    }

    #[test]
    fn q6_layout_rejects_invalid_or_overflowing_geometry() {
        let manifest = validate_manifest(parse_frozen()).expect("validate frozen manifest");
        let mut bad = GrammarLmHeadProfile::A3b.spec();
        bad.hidden = 128;
        assert!(build_bank_layout(&manifest, bad).is_err());
        assert!(checked_align_up(usize::MAX, BANK_ALIGNMENT).is_err());
    }

    #[test]
    fn synthetic_bank_preserves_incidence_order_rows_and_padding() {
        let row_bytes = Q6_BLOCK_BYTES;
        let mut source = vec![0u8; 3 * row_bytes];
        for row in 0..3 {
            for byte in 0..row_bytes {
                source[row * row_bytes + byte] =
                    (row as u8).wrapping_mul(73).wrapping_add(byte as u8);
            }
        }
        let make_state =
            |bank_state_index, topology_state_index, token_ids: Vec<i32>| ValidatedState {
                bank_state_index,
                topology_state_index,
                prefix: vec![bank_state_index as u8],
                token_ids,
            };
        let state0 = make_state(0, 0, vec![0, 2]);
        let state1 = make_state(1, 1, vec![1, 2]);
        let payload = 2 * row_bytes;
        let span = checked_align_up(payload, BANK_ALIGNMENT).unwrap();
        let layout = BankLayout {
            states: vec![
                StateLayout {
                    state: state0,
                    offset: 0,
                    payload_bytes: payload,
                    span_bytes: span,
                },
                StateLayout {
                    state: state1,
                    offset: span,
                    payload_bytes: payload,
                    span_bytes: span,
                },
            ],
            payload_bytes: 4 * row_bytes,
            padding_bytes: 2 * (span - payload),
            span_bytes: 2 * span,
        };
        let profile = ProfileSpec {
            id: "synthetic",
            expected_path: "",
            file_bytes: 0,
            model_sha256: "",
            arch_kind: ArchKind::Dense,
            layers: 0,
            hidden: 256,
            head_bytes: source.len(),
            row_bytes,
            bank_payload_bytes: layout.payload_bytes,
            bank_padding_bytes: layout.padding_bytes,
            bank_span_bytes: layout.span_bytes,
            t0_ms: 1.0,
        };
        let bank = prepare_bank(&layout, &source, profile).expect("prepare synthetic bank");
        assert_eq!(&bank.bytes[0..row_bytes], &source[0..row_bytes]);
        assert_eq!(
            &bank.bytes[row_bytes..2 * row_bytes],
            &source[2 * row_bytes..3 * row_bytes]
        );
        assert_eq!(
            &bank.bytes[span..span + row_bytes],
            &source[row_bytes..2 * row_bytes]
        );
        assert_eq!(
            &bank.bytes[span + row_bytes..span + 2 * row_bytes],
            &source[2 * row_bytes..3 * row_bytes]
        );
        assert!(
            bank.bytes[payload..span]
                .iter()
                .all(|&byte| byte == BANK_POISON)
        );
        assert!(
            bank.bytes[span + payload..2 * span]
                .iter()
                .all(|&byte| byte == BANK_POISON)
        );
        assert_eq!(bank.source_row_hashes.len(), 3);

        assert!(prepare_bank(&layout, &source[..source.len() - 1], profile).is_err());
        let mut out_of_range = layout;
        out_of_range.states[1].state.token_ids[1] = 3;
        assert!(prepare_bank(&out_of_range, &source, profile).is_err());
    }

    #[test]
    fn frozen_gates_do_not_double_apply_equivalent_rounding() {
        let decision = frozen_head_gate_decision(0.82, 0.246, 9.32);
        assert!(decision.head_removal_pass);
        assert!(decision.whole_token_saving_pass);
        assert!(!decision.equivalent_diagnostic);
        assert!(decision.head_removal >= 0.70);
    }

    #[test]
    fn schedule_rotation_and_pair_direction_are_frozen() {
        assert_eq!(ordered_widths(0), WIDTHS);
        assert_eq!(ordered_widths(1), [3, 4, 5, 6, 7, 11, 12, 17, 2]);
        assert_eq!(ordered_widths(9), WIDTHS);
        assert_eq!(arm_order(0), [Arm::Full, Arm::Compact]);
        assert_eq!(arm_order(1), [Arm::Compact, Arm::Full]);
        assert_eq!(arm_order(2), [Arm::Full, Arm::Compact]);
    }
}
