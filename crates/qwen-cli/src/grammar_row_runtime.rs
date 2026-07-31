use crate::grammar_lm_head_row_floor::{ManifestDocument, ValidatedManifest, validate_manifest};
use crate::response_shape_runtime::{
    ResponseShapeRuntime, RuntimeEdge, RuntimeState, RuntimeStateKind,
};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const RUNTIME_FILE_SHA256: &str =
    "69ca90ba2d4519052e4d453fd8e9759ca936d97131f5cb7ff786b2fc33cdeeb5";
const RUNTIME_SEMANTIC_SHA256: &str =
    "2c6da252d62b3df75d0cb31a4a79ea310c3d495f9c3e9d1fcd2509d74175df08";
const MANIFEST_FILE_SHA256: &str =
    "2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0";
const VOCAB_ROWS: usize = 248_320;
const HIDDEN: usize = 2_048;
const Q6_BLOCK_ELEMENTS: usize = 256;
const Q6_BLOCK_BYTES: usize = 210;
const ROW_BYTES: usize = 1_680;
const BANK_ALIGNMENT: usize = 32;
const BANK_POISON: u8 = 0xa5;

const BRANCH_STATES: usize = 451;
const BRANCH_INCIDENCES: usize = 1_468;
const BRANCH_PAYLOAD_BYTES: usize = 2_466_240;
const BRANCH_PADDING_BYTES: usize = 2_720;
const BRANCH_SPAN_BYTES: usize = 2_468_960;

const SINGLETON_STATES: usize = 129;
const SINGLETON_INCIDENCES: usize = 129;
const SINGLETON_PAYLOAD_BYTES: usize = 216_720;
const SINGLETON_PADDING_BYTES: usize = 2_064;
const SINGLETON_SPAN_BYTES: usize = 218_784;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BankKey {
    Branch(u32),
    Singleton(u32),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GrammarState(u32);

impl GrammarState {
    pub(crate) fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StateRef<'a> {
    state: &'a RuntimeState,
    candidates: &'a [RuntimeEdge],
}

impl<'a> StateRef<'a> {
    pub(crate) fn kind(self) -> RuntimeStateKind {
        self.state.kind
    }

    pub(crate) fn candidates(self) -> &'a [RuntimeEdge] {
        self.candidates
    }

    pub(crate) fn bank_key(self) -> Option<BankKey> {
        match (self.state.kind, self.state.bank_state_index) {
            (RuntimeStateKind::Branch, Some(index)) => Some(BankKey::Branch(index)),
            (RuntimeStateKind::Singleton, Some(index)) => Some(BankKey::Singleton(index)),
            (RuntimeStateKind::Terminal, None) => None,
            _ => unreachable!("authenticated runtime state has inconsistent bank metadata"),
        }
    }

    pub(crate) fn is_terminal(self) -> bool {
        self.state.kind == RuntimeStateKind::Terminal
    }
}

#[derive(Debug)]
pub(crate) struct FrozenGrammarRuntime {
    document: ResponseShapeRuntime,
    prefixes: Vec<Vec<u8>>,
    pieces: Vec<Vec<u8>>,
}

impl FrozenGrammarRuntime {
    pub(crate) fn parse_authenticated(bytes: &[u8]) -> Result<Self> {
        ensure!(
            sha256_hex(bytes) == RUNTIME_FILE_SHA256,
            "response-shape runtime complete-file hash mismatch"
        );
        let document: ResponseShapeRuntime =
            serde_json::from_slice(bytes).context("parse response-shape runtime")?;
        document.validate_frozen()?;
        ensure!(
            independent_semantic_sha256(&document)? == RUNTIME_SEMANTIC_SHA256,
            "independent response-shape semantic digest mismatch"
        );
        Self::from_validated_document(document)
    }

    fn from_validated_document(document: ResponseShapeRuntime) -> Result<Self> {
        let prefixes = document
            .states
            .iter()
            .map(|state| decode_hex(&state.prefix_hex))
            .collect::<Result<Vec<_>>>()?;
        let pieces = document
            .edges
            .iter()
            .map(|edge| decode_hex(&edge.piece_hex))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            prefixes.len() == document.states.len() && pieces.len() == document.edges.len(),
            "runtime decode cache length mismatch"
        );

        for (state_index, state) in document.states.iter().enumerate() {
            let start = usize::try_from(state.edge_start)?;
            let end = start
                .checked_add(usize::try_from(state.edge_count)?)
                .context("runtime candidate span overflow")?;
            let candidates = document
                .edges
                .get(start..end)
                .context("runtime candidate span is out of range")?;
            for (local_index, edge) in candidates.iter().enumerate() {
                ensure!(
                    usize::try_from(edge.source_state_index)? == state_index,
                    "runtime candidate source mismatch"
                );
                let successor = usize::try_from(edge.successor_state_index)?;
                let successor_prefix = prefixes
                    .get(successor)
                    .context("runtime candidate successor is out of range")?;
                let mut expected = prefixes[state_index].clone();
                expected.extend_from_slice(&pieces[start + local_index]);
                ensure!(
                    &expected == successor_prefix,
                    "runtime candidate does not reproduce successor prefix"
                );
            }
        }
        Ok(Self {
            document,
            prefixes,
            pieces,
        })
    }

    pub(crate) fn root(&self) -> GrammarState {
        GrammarState(self.document.root_state_index)
    }

    pub(crate) fn state(&self, state: GrammarState) -> Result<StateRef<'_>> {
        let state = self
            .document
            .states
            .get(usize::try_from(state.0)?)
            .context("grammar state is out of range")?;
        let start = usize::try_from(state.edge_start)?;
        let end = start
            .checked_add(usize::try_from(state.edge_count)?)
            .context("grammar candidate span overflow")?;
        let candidates = self
            .document
            .edges
            .get(start..end)
            .context("grammar candidate span is out of range")?;
        Ok(StateRef { state, candidates })
    }

    pub(crate) fn advance(
        &self,
        state: GrammarState,
        token_id: i32,
    ) -> Result<(GrammarState, &[u8])> {
        let state_ref = self.state(state)?;
        ensure!(
            !state_ref.is_terminal(),
            "cannot advance a terminal grammar state"
        );
        let local = state_ref
            .candidates
            .binary_search_by_key(&token_id, |edge| edge.token_id)
            .map_err(|_| anyhow::anyhow!("token is absent from grammar state"))?;
        let edge_start = usize::try_from(state_ref.state.edge_start)?;
        let edge = &state_ref.candidates[local];
        Ok((
            GrammarState(edge.successor_state_index),
            &self.pieces[edge_start + local],
        ))
    }

    pub(crate) fn prefix(&self, state: GrammarState) -> Result<&[u8]> {
        self.prefixes
            .get(usize::try_from(state.0)?)
            .map(Vec::as_slice)
            .context("grammar prefix state is out of range")
    }

    pub(crate) fn document(&self) -> &ResponseShapeRuntime {
        &self.document
    }
}

#[derive(Debug)]
pub(crate) struct FrozenBranchManifest {
    manifest: ValidatedManifest,
}

impl FrozenBranchManifest {
    pub(crate) fn parse_authenticated(bytes: &[u8]) -> Result<Self> {
        ensure!(
            sha256_hex(bytes) == MANIFEST_FILE_SHA256,
            "branch manifest complete-file hash mismatch"
        );
        let document: ManifestDocument =
            serde_json::from_slice(bytes).context("parse branch manifest")?;
        Ok(Self {
            manifest: validate_manifest(document)?,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BankStatePlan {
    pub runtime_state_index: u32,
    pub bank_state_index: u32,
    pub token_ids: Vec<i32>,
    pub offset: usize,
    pub payload_bytes: usize,
    pub span_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct BankPlan {
    pub states: Vec<BankStatePlan>,
    pub row_incidences: usize,
    pub payload_bytes: usize,
    pub padding_bytes: usize,
    pub span_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateBankPlan {
    pub branch: BankPlan,
    pub singleton: BankPlan,
}

impl CandidateBankPlan {
    pub(crate) fn build(
        runtime: &FrozenGrammarRuntime,
        manifest: &FrozenBranchManifest,
    ) -> Result<Self> {
        let mut branch_states = Vec::with_capacity(BRANCH_STATES);
        for manifest_state in manifest.manifest.states() {
            let runtime_index = u32::try_from(manifest_state.topology_state_index())?;
            let runtime_state = runtime.state(GrammarState(runtime_index))?;
            ensure!(
                runtime_state.kind() == RuntimeStateKind::Branch,
                "manifest state does not map to a runtime branch"
            );
            ensure!(
                runtime_state.bank_key()
                    == Some(BankKey::Branch(u32::try_from(
                        manifest_state.bank_state_index()
                    )?)),
                "manifest/runtime branch-bank index mismatch"
            );
            ensure!(
                runtime.prefix(GrammarState(runtime_index))? == manifest_state.prefix(),
                "manifest/runtime branch prefix mismatch"
            );
            let runtime_tokens: Vec<i32> = runtime_state
                .candidates()
                .iter()
                .map(|edge| edge.token_id)
                .collect();
            ensure!(
                runtime_tokens == manifest_state.token_ids(),
                "manifest/runtime branch token mismatch"
            );
            branch_states.push((
                runtime_index,
                u32::try_from(manifest_state.bank_state_index())?,
                runtime_tokens,
            ));
        }

        let singleton_states = runtime
            .document()
            .states
            .iter()
            .filter(|state| state.kind == RuntimeStateKind::Singleton)
            .map(|state| {
                let state_ref = runtime.state(GrammarState(state.state_index))?;
                let bank_state_index = match state_ref.bank_key() {
                    Some(BankKey::Singleton(index)) => index,
                    _ => return Err(anyhow::anyhow!("singleton state has the wrong bank key")),
                };
                Ok((
                    state.state_index,
                    bank_state_index,
                    vec![state_ref.candidates()[0].token_id],
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let branch = build_bank_plan(branch_states)?;
        let singleton = build_bank_plan(singleton_states)?;
        validate_plan_totals(&branch, &singleton)?;
        Ok(Self { branch, singleton })
    }

    pub(crate) fn state(&self, key: BankKey) -> Result<&BankStatePlan> {
        let (states, index) = match key {
            BankKey::Branch(index) => (&self.branch.states, index),
            BankKey::Singleton(index) => (&self.singleton.states, index),
        };
        let state = states
            .get(usize::try_from(index)?)
            .context("bank state index is out of range")?;
        ensure!(
            state.bank_state_index == index,
            "bank plan index does not match position"
        );
        Ok(state)
    }

    pub(crate) fn prepare_from_output_weight(
        &self,
        output_weight: &[u8],
    ) -> Result<PreparedCandidateBanks> {
        let source = A3bOutputWeightRows::new(output_weight)?;
        self.prepare_from_source(&source)
    }

    fn prepare_from_source<S: RowSource>(&self, source: &S) -> Result<PreparedCandidateBanks> {
        let branch = prepare_bank(&self.branch, source)?;
        let singleton = prepare_bank(&self.singleton, source)?;
        Ok(PreparedCandidateBanks { branch, singleton })
    }
}

#[derive(Debug)]
pub(crate) struct PreparedBank {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub source_rows_sha256: String,
    pub row_copies: usize,
    source_row_hashes: BTreeMap<i32, String>,
}

#[derive(Debug)]
pub(crate) struct PreparedCandidateBanks {
    pub branch: PreparedBank,
    pub singleton: PreparedBank,
}

impl PreparedCandidateBanks {
    pub(crate) fn verify_output_weight_unchanged(&self, output_weight: &[u8]) -> Result<()> {
        let source = A3bOutputWeightRows::new(output_weight)?;
        verify_source_rows(&self.branch.source_row_hashes, &source)?;
        verify_source_rows(&self.singleton.source_row_hashes, &source)?;
        Ok(())
    }
}

trait RowSource {
    fn row(&self, token_id: i32) -> Result<&[u8]>;
}

struct A3bOutputWeightRows<'a> {
    bytes: &'a [u8],
}

impl<'a> A3bOutputWeightRows<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        ensure!(
            bytes.len() == VOCAB_ROWS * ROW_BYTES,
            "output.weight source has the wrong byte length"
        );
        Ok(Self { bytes })
    }
}

impl RowSource for A3bOutputWeightRows<'_> {
    fn row(&self, token_id: i32) -> Result<&[u8]> {
        let token = usize::try_from(token_id).context("negative output.weight token row")?;
        ensure!(
            token < VOCAB_ROWS,
            "output.weight token row is out of range"
        );
        let start = token
            .checked_mul(ROW_BYTES)
            .context("output.weight row offset overflow")?;
        let end = start
            .checked_add(ROW_BYTES)
            .context("output.weight row endpoint overflow")?;
        self.bytes
            .get(start..end)
            .context("output.weight row exceeds source bytes")
    }
}

fn build_bank_plan(states: Vec<(u32, u32, Vec<i32>)>) -> Result<BankPlan> {
    ensure!(
        HIDDEN % Q6_BLOCK_ELEMENTS == 0 && HIDDEN / Q6_BLOCK_ELEMENTS * Q6_BLOCK_BYTES == ROW_BYTES,
        "frozen Q6_K row geometry mismatch"
    );
    let mut layouts = Vec::with_capacity(states.len());
    let mut offset = 0usize;
    let mut payload_total = 0usize;
    let mut row_incidences = 0usize;
    for (position, (runtime_state_index, bank_state_index, token_ids)) in
        states.into_iter().enumerate()
    {
        ensure!(
            usize::try_from(bank_state_index)? == position,
            "bank state indices are not contiguous"
        );
        ensure!(!token_ids.is_empty(), "bank state has no token rows");
        ensure!(
            token_ids.windows(2).all(|pair| pair[0] < pair[1])
                && token_ids
                    .iter()
                    .all(|&token| token >= 0 && token < VOCAB_ROWS as i32),
            "bank token rows are not canonical"
        );
        let payload_bytes = token_ids
            .len()
            .checked_mul(ROW_BYTES)
            .context("bank state payload overflow")?;
        let span_bytes = checked_align_up(payload_bytes, BANK_ALIGNMENT)?;
        ensure!(offset % BANK_ALIGNMENT == 0, "bank state is misaligned");
        layouts.push(BankStatePlan {
            runtime_state_index,
            bank_state_index,
            token_ids,
            offset,
            payload_bytes,
            span_bytes,
        });
        row_incidences = row_incidences
            .checked_add(layouts.last().expect("layout was pushed").token_ids.len())
            .context("bank row-incidence overflow")?;
        payload_total = payload_total
            .checked_add(payload_bytes)
            .context("bank payload overflow")?;
        offset = offset
            .checked_add(span_bytes)
            .context("bank span overflow")?;
    }
    Ok(BankPlan {
        states: layouts,
        row_incidences,
        payload_bytes: payload_total,
        padding_bytes: offset
            .checked_sub(payload_total)
            .context("bank padding underflow")?,
        span_bytes: offset,
    })
}

fn validate_plan_totals(branch: &BankPlan, singleton: &BankPlan) -> Result<()> {
    ensure!(
        branch.states.len() == BRANCH_STATES
            && branch.row_incidences == BRANCH_INCIDENCES
            && branch.payload_bytes == BRANCH_PAYLOAD_BYTES
            && branch.padding_bytes == BRANCH_PADDING_BYTES
            && branch.span_bytes == BRANCH_SPAN_BYTES,
        "branch bank totals differ from v0.655"
    );
    ensure!(
        singleton.states.len() == SINGLETON_STATES
            && singleton.row_incidences == SINGLETON_INCIDENCES
            && singleton.payload_bytes == SINGLETON_PAYLOAD_BYTES
            && singleton.padding_bytes == SINGLETON_PADDING_BYTES
            && singleton.span_bytes == SINGLETON_SPAN_BYTES,
        "singleton bank totals differ from the frozen runtime"
    );
    ensure!(
        branch.row_incidences + singleton.row_incidences == 1_597
            && branch.payload_bytes + singleton.payload_bytes == 2_682_960
            && branch.span_bytes + singleton.span_bytes == 2_687_744,
        "combined candidate bank totals mismatch"
    );
    Ok(())
}

fn prepare_bank<S: RowSource>(plan: &BankPlan, source: &S) -> Result<PreparedBank> {
    let mut bytes = vec![BANK_POISON; plan.span_bytes];
    let mut source_rows = BTreeMap::new();
    let mut row_copies = 0usize;
    for state in &plan.states {
        let mut destination = state.offset;
        for &token_id in &state.token_ids {
            let end = destination
                .checked_add(ROW_BYTES)
                .context("bank row endpoint overflow")?;
            let row = bytes
                .get_mut(destination..end)
                .context("bank row exceeds destination")?;
            let source_row = source.row(token_id)?;
            ensure!(
                source_row.len() == ROW_BYTES,
                "output.weight source row has the wrong byte length"
            );
            row.copy_from_slice(source_row);
            ensure!(
                row == source_row,
                "bank row differs from output.weight source"
            );
            row_copies = row_copies.checked_add(1).context("row-copy overflow")?;
            let digest = sha256_hex(source_row);
            if let Some(existing) = source_rows.insert(token_id, digest.clone()) {
                ensure!(existing == digest, "repeated source row changed");
            }
            destination = end;
        }
        ensure!(
            destination == state.offset + state.payload_bytes,
            "bank rows do not cover state payload"
        );
        ensure!(
            bytes[state.offset + state.payload_bytes..state.offset + state.span_bytes]
                .iter()
                .all(|&byte| byte == BANK_POISON),
            "bank padding poison changed"
        );
    }
    Ok(PreparedBank {
        sha256: sha256_hex(&bytes),
        source_rows_sha256: source_row_set_digest(&source_rows)?,
        row_copies,
        source_row_hashes: source_rows,
        bytes,
    })
}

fn verify_source_rows<S: RowSource>(expected: &BTreeMap<i32, String>, source: &S) -> Result<()> {
    for (&token_id, expected_digest) in expected {
        let row = source.row(token_id)?;
        ensure!(
            row.len() == ROW_BYTES && sha256_hex(row) == *expected_digest,
            "output.weight source row changed after bank construction"
        );
    }
    Ok(())
}

fn source_row_set_digest(rows: &BTreeMap<i32, String>) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"grammar-source-row-set/v1\0");
    for (&token, digest) in rows {
        let raw = decode_hex(digest)?;
        ensure!(raw.len() == 32, "source row digest has the wrong size");
        hasher.update(token.to_le_bytes());
        hasher.update((raw.len() as u64).to_le_bytes());
        hasher.update(raw);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn checked_align_up(value: usize, alignment: usize) -> Result<usize> {
    ensure!(alignment.is_power_of_two(), "bank alignment is invalid");
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("bank alignment overflow")
}

fn independent_semantic_sha256(runtime: &ResponseShapeRuntime) -> Result<String> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"response-shape-runtime/v1\0");
    push_u32(&mut bytes, runtime.vocab_rows);
    push_u32(&mut bytes, runtime.root_state_index);
    push_len(&mut bytes, runtime.stop_token_ids.len())?;
    for &token in &runtime.stop_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }

    for (name, digest) in [
        (
            "grammar_file_sha256",
            &runtime.fingerprints.grammar_file_sha256,
        ),
        ("trace_file_sha256", &runtime.fingerprints.trace_file_sha256),
        (
            "branch_manifest_file_sha256",
            &runtime.fingerprints.branch_manifest_file_sha256,
        ),
        (
            "grammar_piece_policy_sha256",
            &runtime.fingerprints.grammar_piece_policy_sha256,
        ),
        (
            "finite_language_sha256",
            &runtime.fingerprints.finite_language_sha256,
        ),
        ("state_sha256", &runtime.fingerprints.state_sha256),
        (
            "canonical_path_sha256",
            &runtime.fingerprints.canonical_path_sha256,
        ),
    ] {
        push_bytes(&mut bytes, name.as_bytes())?;
        let digest = decode_hex(digest)?;
        ensure!(digest.len() == 32, "runtime fingerprint has the wrong size");
        bytes.extend_from_slice(&digest);
    }

    push_len(&mut bytes, runtime.states.len())?;
    for state in &runtime.states {
        push_u32(&mut bytes, state.state_index);
        bytes.push(match state.kind {
            RuntimeStateKind::Branch => 1,
            RuntimeStateKind::Singleton => 2,
            RuntimeStateKind::Terminal => 3,
        });
        push_bytes(&mut bytes, &decode_hex(&state.prefix_hex)?)?;
        match state.bank_state_index {
            None => bytes.push(0),
            Some(index) => {
                bytes.push(1);
                push_u32(&mut bytes, index);
            }
        }
        push_u32(&mut bytes, state.edge_start);
        push_u32(&mut bytes, state.edge_count);
    }

    push_len(&mut bytes, runtime.edges.len())?;
    for edge in &runtime.edges {
        push_u32(&mut bytes, edge.source_state_index);
        bytes.extend_from_slice(&edge.token_id.to_le_bytes());
        push_bytes(&mut bytes, &decode_hex(&edge.piece_hex)?)?;
        push_u32(&mut bytes, edge.successor_state_index);
    }

    push_len(&mut bytes, runtime.terminal_state_indices.len())?;
    for &state in &runtime.terminal_state_indices {
        push_u32(&mut bytes, state);
    }

    push_len(&mut bytes, runtime.canonical_paths.len())?;
    for path in &runtime.canonical_paths {
        push_u32(&mut bytes, path.path_index);
        push_bytes(&mut bytes, &decode_hex(&path.target_hex)?)?;
        let target_sha256 = decode_hex(&path.target_sha256)?;
        ensure!(
            target_sha256.len() == 32,
            "target digest has the wrong size"
        );
        bytes.extend_from_slice(&target_sha256);
        push_len(&mut bytes, path.token_ids.len())?;
        for &token in &path.token_ids {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        push_len(&mut bytes, path.state_indices.len())?;
        for &state in &path.state_indices {
            push_u32(&mut bytes, state);
        }
        push_u32(&mut bytes, path.terminal_state_index);
    }
    push_u32(
        &mut bytes,
        runtime.path_bounds.minimum_root_to_terminal_tokens,
    );
    push_u32(
        &mut bytes,
        runtime.path_bounds.maximum_root_to_terminal_tokens,
    );
    Ok(sha256_hex(&bytes))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_len(bytes: &mut Vec<u8>, length: usize) -> Result<()> {
    bytes.extend_from_slice(&u64::try_from(length)?.to_le_bytes());
    Ok(())
}

fn push_bytes(bytes: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    push_len(bytes, value.len())?;
    bytes.extend_from_slice(value);
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    ensure!(value.len() % 2 == 0, "hex string has odd length");
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "hex string is not canonical lowercase hexadecimal"
    );
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).context("hex pair is not UTF-8")?;
            u8::from_str_radix(text, 16).context("parse hex pair")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const RUNTIME_BYTES: &[u8] =
        include_bytes!("../../../docs/bench/v0656-response-shape-runtime.json");
    const MANIFEST_BYTES: &[u8] =
        include_bytes!("../../../docs/bench/v0655-grammar-branch-manifest.json");

    fn fixtures() -> (
        FrozenGrammarRuntime,
        FrozenBranchManifest,
        CandidateBankPlan,
    ) {
        let runtime = FrozenGrammarRuntime::parse_authenticated(RUNTIME_BYTES).unwrap();
        let manifest = FrozenBranchManifest::parse_authenticated(MANIFEST_BYTES).unwrap();
        let plan = CandidateBankPlan::build(&runtime, &manifest).unwrap();
        (runtime, manifest, plan)
    }

    struct SyntheticRows {
        rows: BTreeMap<i32, Vec<u8>>,
        calls: RefCell<BTreeMap<i32, usize>>,
    }

    impl SyntheticRows {
        fn for_plan(plan: &CandidateBankPlan) -> Self {
            let mut rows = BTreeMap::new();
            for token_id in plan
                .branch
                .states
                .iter()
                .chain(&plan.singleton.states)
                .flat_map(|state| state.token_ids.iter().copied())
            {
                rows.entry(token_id).or_insert_with(|| {
                    let mut row = vec![0u8; ROW_BYTES];
                    row[..4].copy_from_slice(&token_id.to_le_bytes());
                    for (index, byte) in row[4..].iter_mut().enumerate() {
                        *byte = (token_id as u32)
                            .wrapping_mul(31)
                            .wrapping_add(index as u32) as u8;
                    }
                    row
                });
            }
            Self {
                rows,
                calls: RefCell::new(BTreeMap::new()),
            }
        }
    }

    impl RowSource for SyntheticRows {
        fn row(&self, token_id: i32) -> Result<&[u8]> {
            *self.calls.borrow_mut().entry(token_id).or_default() += 1;
            self.rows
                .get(&token_id)
                .map(Vec::as_slice)
                .context("synthetic source row is absent")
        }
    }

    #[test]
    fn authenticated_artifacts_cross_bind_all_states() {
        let (runtime, _manifest, plan) = fixtures();
        assert_eq!(runtime.root().index(), 0);
        assert_eq!(plan.branch.states.len(), BRANCH_STATES);
        assert_eq!(plan.singleton.states.len(), SINGLETON_STATES);
        assert_eq!(plan.branch.row_incidences, BRANCH_INCIDENCES);
        assert_eq!(plan.singleton.row_incidences, SINGLETON_INCIDENCES);
        assert_eq!(plan.branch.span_bytes, BRANCH_SPAN_BYTES);
        assert_eq!(plan.singleton.span_bytes, SINGLETON_SPAN_BYTES);

        let mut edge_count = 0usize;
        for state_index in 0..runtime.document().states.len() {
            let state = GrammarState(u32::try_from(state_index).unwrap());
            let state_ref = runtime.state(state).unwrap();
            for edge in state_ref.candidates() {
                let (successor, piece) = runtime.advance(state, edge.token_id).unwrap();
                assert_eq!(successor.index(), edge.successor_state_index);
                assert!(!piece.is_empty());
                edge_count += 1;
            }
            match state_ref.kind() {
                RuntimeStateKind::Branch => {
                    assert!(matches!(state_ref.bank_key(), Some(BankKey::Branch(_))));
                }
                RuntimeStateKind::Singleton => {
                    assert!(matches!(state_ref.bank_key(), Some(BankKey::Singleton(_))));
                }
                RuntimeStateKind::Terminal => assert_eq!(state_ref.bank_key(), None),
            }
        }
        assert_eq!(edge_count, 1_597);

        let first_branch = plan.state(BankKey::Branch(0)).unwrap();
        let first_singleton = plan.state(BankKey::Singleton(0)).unwrap();
        assert_eq!(first_branch.bank_state_index, 0);
        assert_eq!(first_branch.runtime_state_index, 0);
        assert_eq!(first_singleton.bank_state_index, 0);
        assert_eq!(first_singleton.runtime_state_index, 1);
    }

    #[test]
    fn canonical_paths_replay_through_runtime_state_api() {
        let (runtime, _manifest, _plan) = fixtures();
        for path in &runtime.document().canonical_paths {
            let mut state = runtime.root();
            let mut bytes = Vec::new();
            assert_eq!(path.state_indices.first(), Some(&state.index()));
            for (step, &token_id) in path.token_ids.iter().enumerate() {
                let (successor, piece) = runtime.advance(state, token_id).unwrap();
                bytes.extend_from_slice(piece);
                assert_eq!(successor.index(), path.state_indices[step + 1]);
                state = successor;
            }
            assert_eq!(bytes, decode_hex(&path.target_hex).unwrap());
            assert_eq!(state.index(), path.terminal_state_index);
            assert!(runtime.state(state).unwrap().is_terminal());
            assert_eq!(runtime.prefix(state).unwrap(), bytes);
        }
    }

    #[test]
    fn runtime_state_api_rejects_invalid_transitions() {
        let (runtime, _manifest, _plan) = fixtures();
        assert!(runtime.state(GrammarState(u32::MAX)).is_err());
        assert!(runtime.advance(runtime.root(), -1).is_err());
        let terminal = GrammarState(runtime.document().terminal_state_indices[0]);
        assert!(runtime.advance(terminal, 0).is_err());
    }

    #[test]
    fn semantic_validator_rejects_foundational_graph_mutations() {
        let base: ResponseShapeRuntime = serde_json::from_slice(RUNTIME_BYTES).unwrap();
        let reject = |mutated: &mut ResponseShapeRuntime| {
            mutated.semantic_runtime_table_sha256 = mutated.semantic_sha256().unwrap();
            assert!(mutated.validate_frozen().is_err());
        };

        let mut wrong_count = base.clone();
        wrong_count.counts.edges += 1;
        reject(&mut wrong_count);

        let mut bad_span = base.clone();
        bad_span.states[0].edge_count += 1;
        reject(&mut bad_span);

        let mut duplicate_edge = base.clone();
        duplicate_edge.edges[1].token_id = duplicate_edge.edges[0].token_id;
        reject(&mut duplicate_edge);

        let mut unsorted_edges = base.clone();
        unsorted_edges.edges.swap(0, 1);
        reject(&mut unsorted_edges);

        let mut bad_successor = base.clone();
        bad_successor.edges[0].successor_state_index = 0;
        reject(&mut bad_successor);

        let mut out_of_range_successor = base.clone();
        out_of_range_successor.edges[0].successor_state_index = u32::MAX;
        reject(&mut out_of_range_successor);

        let mut stop_edge = base.clone();
        stop_edge.edges[0].token_id = 248_046;
        reject(&mut stop_edge);

        let mut wrong_bounds = base;
        wrong_bounds.path_bounds.maximum_root_to_terminal_tokens -= 1;
        reject(&mut wrong_bounds);
    }

    #[test]
    fn candidate_bank_preparation_preserves_all_padding() {
        let (_runtime, _manifest, plan) = fixtures();
        let source = SyntheticRows::for_plan(&plan);
        let prepared = plan.prepare_from_source(&source).unwrap();
        assert_eq!(prepared.branch.bytes.len(), BRANCH_SPAN_BYTES);
        assert_eq!(prepared.singleton.bytes.len(), SINGLETON_SPAN_BYTES);
        assert_eq!(prepared.branch.row_copies, BRANCH_INCIDENCES);
        assert_eq!(prepared.singleton.row_copies, SINGLETON_INCIDENCES);
        assert_eq!(source.calls.borrow().values().sum::<usize>(), 1_597);
        assert_eq!(prepared.branch.sha256.len(), 64);
        assert_eq!(prepared.singleton.sha256.len(), 64);
        assert_eq!(prepared.branch.source_rows_sha256.len(), 64);
        assert_eq!(prepared.singleton.source_rows_sha256.len(), 64);
        assert!(
            plan.branch
                .states
                .iter()
                .chain(&plan.singleton.states)
                .all(|state| state.offset % BANK_ALIGNMENT == 0
                    && state.span_bytes % BANK_ALIGNMENT == 0)
        );

        for (bank_plan, prepared_bank) in [
            (&plan.branch, &prepared.branch),
            (&plan.singleton, &prepared.singleton),
        ] {
            for state in &bank_plan.states {
                for (local, token_id) in state.token_ids.iter().enumerate() {
                    let start = state.offset + local * ROW_BYTES;
                    let source_row = source.rows.get(token_id).unwrap();
                    assert_eq!(&prepared_bank.bytes[start..start + ROW_BYTES], source_row);
                }
                assert!(
                    prepared_bank.bytes
                        [state.offset + state.payload_bytes..state.offset + state.span_bytes]
                        .iter()
                        .all(|&byte| byte == BANK_POISON)
                );
            }
            assert_eq!(prepared_bank.sha256, sha256_hex(&prepared_bank.bytes));
            assert_eq!(
                prepared_bank.source_rows_sha256,
                source_row_set_digest(&prepared_bank.source_row_hashes).unwrap()
            );
        }

        let mut expected_calls = BTreeMap::new();
        for token_id in plan
            .branch
            .states
            .iter()
            .chain(&plan.singleton.states)
            .flat_map(|state| state.token_ids.iter().copied())
        {
            *expected_calls.entry(token_id).or_default() += 1;
        }
        assert_eq!(*source.calls.borrow(), expected_calls);
        let mut singleton_frequency = BTreeMap::new();
        for token_id in plan
            .singleton
            .states
            .iter()
            .flat_map(|state| state.token_ids.iter().copied())
        {
            *singleton_frequency.entry(token_id).or_default() += 1;
        }
        assert_eq!(singleton_frequency.len(), 12);
        assert_eq!(singleton_frequency.values().copied().max(), Some(36));
    }

    #[test]
    fn source_bound_bank_preparation_rejects_bad_sources() {
        assert!(A3bOutputWeightRows::new(&[0u8; ROW_BYTES]).is_err());
        let (_runtime, _manifest, plan) = fixtures();
        let mut source = SyntheticRows::for_plan(&plan);
        let missing = plan.branch.states[0].token_ids[0];
        source.rows.remove(&missing);
        assert!(plan.prepare_from_source(&source).is_err());

        struct ShortRow(Vec<u8>);
        impl RowSource for ShortRow {
            fn row(&self, _token_id: i32) -> Result<&[u8]> {
                Ok(&self.0)
            }
        }
        assert!(prepare_bank(&plan.singleton, &ShortRow(vec![0; ROW_BYTES - 1])).is_err());
    }

    #[test]
    fn authenticated_parsers_reject_byte_changes() {
        let mut runtime = RUNTIME_BYTES.to_vec();
        runtime.push(b' ');
        assert!(FrozenGrammarRuntime::parse_authenticated(&runtime).is_err());
        let mut manifest = MANIFEST_BYTES.to_vec();
        manifest.push(b' ');
        assert!(FrozenBranchManifest::parse_authenticated(&manifest).is_err());
    }
}
