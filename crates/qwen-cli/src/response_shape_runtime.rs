use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeSet, VecDeque};

pub(crate) const SCHEMA: &str = "response-shape-runtime/v1";
pub(crate) const CLAIM_SCOPE: &str =
    "compiled exact response-shape table; no general grammar or performance authority";

const VOCAB_ROWS: u32 = 248_320;
const STATE_COUNT: usize = 616;
const BRANCH_STATE_COUNT: usize = 451;
const SINGLETON_STATE_COUNT: usize = 129;
const TERMINAL_STATE_COUNT: usize = 36;
const EDGE_COUNT: usize = 1_597;
const CANONICAL_PATH_COUNT: usize = 36;
const ROOT_STATE_INDEX: u32 = 0;
const STOP_TOKEN_IDS: [i32; 1] = [248_046];
const CANONICAL_PATH_TOKEN_COUNT: u32 = 18;
const MINIMUM_PATH_TOKEN_COUNT: u32 = 18;
const MAXIMUM_PATH_TOKEN_COUNT: u32 = 84;

const GRAMMAR_FILE_SHA256: &str =
    "f3a053733738aa8e05c10c2f078ca5a36c26a3bc5746487903463cdcf31b2634";
const TRACE_FILE_SHA256: &str = "eb702b6b93fe58255cc16783363150f886dbfd0d6f7c032c1798bae1c41b74b1";
const BRANCH_MANIFEST_FILE_SHA256: &str =
    "2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0";
const GRAMMAR_PIECE_POLICY_SHA256: &str =
    "58748dfe811c1710c732239ae59053f2888bf32c1c3ee4f8ec10f0ac089fdd60";
const FINITE_LANGUAGE_SHA256: &str =
    "94dc537ecb3d9209cfbc8e28b502916b20ee7b685393b6cf85ab610c08be0faf";
const STATE_SHA256: &str = "afa64a9a91677309fffbc54833b73d8881d7c4745127ae5ec10eebc87d70be40";
const CANONICAL_PATH_SHA256: &str =
    "b72213294b55231dd4eaa12a05276be7dfe4e5ce994ff7a9451c258876333401";

const FINGERPRINT_NAMES: [&str; 7] = [
    "grammar_file_sha256",
    "trace_file_sha256",
    "branch_manifest_file_sha256",
    "grammar_piece_policy_sha256",
    "finite_language_sha256",
    "state_sha256",
    "canonical_path_sha256",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponseShapeRuntime {
    pub schema: String,
    pub claim_scope: String,
    pub grammar_id: String,
    pub vocab_rows: u32,
    pub root_state_index: u32,
    pub stop_token_ids: Vec<i32>,
    pub fingerprints: RuntimeFingerprints,
    pub counts: RuntimeCounts,
    pub states: Vec<RuntimeState>,
    pub edges: Vec<RuntimeEdge>,
    pub terminal_state_indices: Vec<u32>,
    pub canonical_paths: Vec<RuntimeCanonicalPath>,
    pub path_bounds: RuntimePathBounds,
    pub semantic_runtime_table_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeFingerprints {
    pub grammar_file_sha256: String,
    pub trace_file_sha256: String,
    pub branch_manifest_file_sha256: String,
    pub grammar_piece_policy_sha256: String,
    pub finite_language_sha256: String,
    pub state_sha256: String,
    pub canonical_path_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeCounts {
    pub states: u32,
    pub branch_states: u32,
    pub singleton_states: u32,
    pub terminal_states: u32,
    pub edges: u32,
    pub canonical_paths: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RuntimeStateKind {
    Branch,
    Singleton,
    Terminal,
}

impl RuntimeStateKind {
    fn semantic_tag(self) -> u8 {
        match self {
            Self::Branch => 1,
            Self::Singleton => 2,
            Self::Terminal => 3,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeState {
    pub state_index: u32,
    pub kind: RuntimeStateKind,
    pub prefix_bytes: u32,
    pub prefix_hex: String,
    /// Index in the branch or singleton bank selected by `kind`.
    pub bank_state_index: Option<u32>,
    pub edge_start: u32,
    pub edge_count: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeEdge {
    pub source_state_index: u32,
    pub token_id: i32,
    pub piece_bytes: u32,
    pub piece_hex: String,
    pub successor_state_index: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeCanonicalPath {
    pub path_index: u32,
    pub target_bytes: u32,
    pub target_hex: String,
    pub target_sha256: String,
    pub token_ids: Vec<i32>,
    pub state_indices: Vec<u32>,
    pub terminal_state_index: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimePathBounds {
    pub minimum_root_to_terminal_tokens: u32,
    pub maximum_root_to_terminal_tokens: u32,
}

impl RuntimeFingerprints {
    fn ordered(&self) -> [(&'static str, &str); 7] {
        [
            (FINGERPRINT_NAMES[0], &self.grammar_file_sha256),
            (FINGERPRINT_NAMES[1], &self.trace_file_sha256),
            (FINGERPRINT_NAMES[2], &self.branch_manifest_file_sha256),
            (FINGERPRINT_NAMES[3], &self.grammar_piece_policy_sha256),
            (FINGERPRINT_NAMES[4], &self.finite_language_sha256),
            (FINGERPRINT_NAMES[5], &self.state_sha256),
            (FINGERPRINT_NAMES[6], &self.canonical_path_sha256),
        ]
    }
}

impl ResponseShapeRuntime {
    #[allow(dead_code)]
    pub(crate) fn seal(mut self) -> Result<Self> {
        ensure!(
            self.semantic_runtime_table_sha256.is_empty(),
            "runtime table must be unsealed before digest construction"
        );
        self.semantic_runtime_table_sha256 = self.semantic_sha256()?;
        self.validate_frozen()?;
        Ok(self)
    }

    pub(crate) fn semantic_sha256(&self) -> Result<String> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"response-shape-runtime/v1\0");
        push_u32(&mut bytes, self.vocab_rows);
        push_u32(&mut bytes, self.root_state_index);

        push_len(&mut bytes, self.stop_token_ids.len())?;
        for &token in &self.stop_token_ids {
            bytes.extend_from_slice(&token.to_le_bytes());
        }

        for (name, digest) in self.fingerprints.ordered() {
            push_bytes(&mut bytes, name.as_bytes())?;
            bytes.extend_from_slice(&decode_sha256(digest)?);
        }

        push_len(&mut bytes, self.states.len())?;
        for state in &self.states {
            push_u32(&mut bytes, state.state_index);
            bytes.push(state.kind.semantic_tag());
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

        push_len(&mut bytes, self.edges.len())?;
        for edge in &self.edges {
            push_u32(&mut bytes, edge.source_state_index);
            bytes.extend_from_slice(&edge.token_id.to_le_bytes());
            push_bytes(&mut bytes, &decode_hex(&edge.piece_hex)?)?;
            push_u32(&mut bytes, edge.successor_state_index);
        }

        push_len(&mut bytes, self.terminal_state_indices.len())?;
        for &state in &self.terminal_state_indices {
            push_u32(&mut bytes, state);
        }

        push_len(&mut bytes, self.canonical_paths.len())?;
        for path in &self.canonical_paths {
            push_u32(&mut bytes, path.path_index);
            push_bytes(&mut bytes, &decode_hex(&path.target_hex)?)?;
            bytes.extend_from_slice(&decode_sha256(&path.target_sha256)?);
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

        push_u32(&mut bytes, self.path_bounds.minimum_root_to_terminal_tokens);
        push_u32(&mut bytes, self.path_bounds.maximum_root_to_terminal_tokens);
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub(crate) fn validate_frozen(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unexpected runtime schema");
        ensure!(self.claim_scope == CLAIM_SCOPE, "unexpected claim scope");
        ensure!(
            self.grammar_id == "response-shape-v1",
            "unexpected grammar id"
        );
        ensure!(self.vocab_rows == VOCAB_ROWS, "unexpected vocabulary size");
        ensure!(
            self.root_state_index == ROOT_STATE_INDEX,
            "unexpected root state"
        );
        ensure!(
            self.stop_token_ids == STOP_TOKEN_IDS,
            "unexpected stop-token set"
        );
        self.validate_fingerprints()?;
        self.validate_counts()?;

        let prefixes = self.validate_states_and_edges()?;
        self.validate_graph(&prefixes)?;
        self.validate_paths(&prefixes)?;

        let derived_bounds = derive_path_bounds(&self.states, &self.edges)?;
        ensure!(
            self.path_bounds == derived_bounds,
            "recorded path bounds do not match graph"
        );
        ensure!(
            derived_bounds.minimum_root_to_terminal_tokens == MINIMUM_PATH_TOKEN_COUNT
                && derived_bounds.maximum_root_to_terminal_tokens == MAXIMUM_PATH_TOKEN_COUNT,
            "unexpected frozen path bounds: {:?}",
            derived_bounds
        );

        ensure!(
            self.semantic_runtime_table_sha256 == self.semantic_sha256()?,
            "runtime semantic digest mismatch"
        );
        Ok(())
    }

    fn validate_fingerprints(&self) -> Result<()> {
        let expected = [
            GRAMMAR_FILE_SHA256,
            TRACE_FILE_SHA256,
            BRANCH_MANIFEST_FILE_SHA256,
            GRAMMAR_PIECE_POLICY_SHA256,
            FINITE_LANGUAGE_SHA256,
            STATE_SHA256,
            CANONICAL_PATH_SHA256,
        ];
        for ((name, actual), expected) in self.fingerprints.ordered().into_iter().zip(expected) {
            decode_sha256(actual).with_context(|| format!("invalid fingerprint {name}"))?;
            ensure!(actual == expected, "unexpected fingerprint {name}");
        }
        Ok(())
    }

    fn validate_counts(&self) -> Result<()> {
        ensure!(self.states.len() == STATE_COUNT, "unexpected state count");
        ensure!(self.edges.len() == EDGE_COUNT, "unexpected edge count");
        ensure!(
            self.terminal_state_indices.len() == TERMINAL_STATE_COUNT,
            "unexpected terminal count"
        );
        ensure!(
            self.canonical_paths.len() == CANONICAL_PATH_COUNT,
            "unexpected canonical path count"
        );
        ensure!(
            self.counts
                == RuntimeCounts {
                    states: STATE_COUNT as u32,
                    branch_states: BRANCH_STATE_COUNT as u32,
                    singleton_states: SINGLETON_STATE_COUNT as u32,
                    terminal_states: TERMINAL_STATE_COUNT as u32,
                    edges: EDGE_COUNT as u32,
                    canonical_paths: CANONICAL_PATH_COUNT as u32,
                },
            "runtime count summary mismatch"
        );
        Ok(())
    }

    fn validate_states_and_edges(&self) -> Result<Vec<Vec<u8>>> {
        let mut prefixes = Vec::with_capacity(self.states.len());
        let mut edge_cursor = 0usize;
        let mut next_branch = 0u32;
        let mut next_singleton = 0u32;
        let mut terminals = Vec::new();

        for (index, state) in self.states.iter().enumerate() {
            ensure!(
                usize::try_from(state.state_index)? == index,
                "runtime state indices are not contiguous"
            );
            let prefix = decode_hex(&state.prefix_hex)?;
            ensure!(
                usize::try_from(state.prefix_bytes)? == prefix.len(),
                "runtime state prefix length mismatch"
            );
            if let Some(previous) = prefixes.last() {
                ensure!(
                    previous < &prefix,
                    "runtime prefixes are not strictly sorted"
                );
            }
            ensure!(
                usize::try_from(state.edge_start)? == edge_cursor,
                "runtime edge spans are not contiguous"
            );
            let edge_count = usize::try_from(state.edge_count)?;
            let edge_end = edge_cursor
                .checked_add(edge_count)
                .context("runtime edge span overflow")?;
            ensure!(
                edge_end <= self.edges.len(),
                "runtime edge span is out of range"
            );

            match state.kind {
                RuntimeStateKind::Branch => {
                    ensure!(edge_count > 1, "branch state has fewer than two edges");
                    ensure!(
                        state.bank_state_index == Some(next_branch),
                        "branch bank indices are not contiguous"
                    );
                    next_branch += 1;
                }
                RuntimeStateKind::Singleton => {
                    ensure!(edge_count == 1, "singleton state does not have one edge");
                    ensure!(
                        state.bank_state_index == Some(next_singleton),
                        "singleton bank indices are not contiguous"
                    );
                    next_singleton += 1;
                }
                RuntimeStateKind::Terminal => {
                    ensure!(edge_count == 0, "terminal state has outgoing edges");
                    ensure!(
                        state.bank_state_index.is_none(),
                        "terminal state has a bank index"
                    );
                    terminals.push(state.state_index);
                }
            }

            let mut previous_token = None;
            for edge in &self.edges[edge_cursor..edge_end] {
                ensure!(
                    edge.source_state_index == state.state_index,
                    "edge source does not match its state span"
                );
                ensure!(
                    edge.token_id >= 0 && u32::try_from(edge.token_id)? < self.vocab_rows,
                    "edge token is out of range"
                );
                ensure!(
                    self.stop_token_ids.binary_search(&edge.token_id).is_err(),
                    "stop token appears in runtime edge"
                );
                if let Some(previous) = previous_token {
                    ensure!(previous < edge.token_id, "state edges are not token sorted");
                }
                previous_token = Some(edge.token_id);
                let piece = decode_hex(&edge.piece_hex)?;
                ensure!(!piece.is_empty(), "runtime edge has an empty piece");
                ensure!(
                    usize::try_from(edge.piece_bytes)? == piece.len(),
                    "runtime edge piece length mismatch"
                );
                let successor = usize::try_from(edge.successor_state_index)?;
                ensure!(
                    successor < self.states.len(),
                    "edge successor is out of range"
                );
            }

            edge_cursor = edge_end;
            prefixes.push(prefix);
        }

        ensure!(
            edge_cursor == self.edges.len(),
            "runtime edges are not covered"
        );
        ensure!(
            next_branch == BRANCH_STATE_COUNT as u32,
            "branch bank namespace has the wrong size"
        );
        ensure!(
            next_singleton == SINGLETON_STATE_COUNT as u32,
            "singleton bank namespace has the wrong size"
        );
        ensure!(
            terminals == self.terminal_state_indices,
            "terminal-state index list mismatch"
        );
        ensure!(
            prefixes[self.root_state_index as usize].is_empty(),
            "root state is not the empty prefix"
        );

        for (state_index, state) in self.states.iter().enumerate() {
            let start = usize::try_from(state.edge_start)?;
            let end = start + usize::try_from(state.edge_count)?;
            for edge in &self.edges[start..end] {
                let piece = decode_hex(&edge.piece_hex)?;
                let mut expected = prefixes[state_index].clone();
                expected.extend_from_slice(&piece);
                let successor = usize::try_from(edge.successor_state_index)?;
                ensure!(
                    expected == prefixes[successor],
                    "edge piece does not reproduce successor prefix"
                );
                ensure!(
                    prefixes[successor].len() > prefixes[state_index].len(),
                    "runtime edge does not increase prefix length"
                );
            }
        }
        Ok(prefixes)
    }

    fn validate_graph(&self, prefixes: &[Vec<u8>]) -> Result<()> {
        let mut reached = vec![false; self.states.len()];
        let root = usize::try_from(self.root_state_index)?;
        reached[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(state_index) = queue.pop_front() {
            let state = &self.states[state_index];
            let start = usize::try_from(state.edge_start)?;
            let end = start + usize::try_from(state.edge_count)?;
            for edge in &self.edges[start..end] {
                let successor = usize::try_from(edge.successor_state_index)?;
                if !reached[successor] {
                    reached[successor] = true;
                    queue.push_back(successor);
                }
            }
        }
        ensure!(
            reached.into_iter().all(|value| value),
            "runtime graph contains an unreachable state"
        );

        let terminal_prefixes: BTreeSet<&[u8]> = self
            .terminal_state_indices
            .iter()
            .map(|&index| prefixes[index as usize].as_slice())
            .collect();
        ensure!(
            terminal_prefixes.len() == TERMINAL_STATE_COUNT,
            "terminal prefixes are not unique"
        );
        Ok(())
    }

    fn validate_paths(&self, prefixes: &[Vec<u8>]) -> Result<()> {
        let mut path_terminals = BTreeSet::new();
        for (path_index, path) in self.canonical_paths.iter().enumerate() {
            ensure!(
                usize::try_from(path.path_index)? == path_index,
                "canonical path indices are not contiguous"
            );
            let target = decode_hex(&path.target_hex)?;
            ensure!(
                usize::try_from(path.target_bytes)? == target.len(),
                "canonical target length mismatch"
            );
            ensure!(
                format!("{:x}", Sha256::digest(&target)) == path.target_sha256,
                "canonical target digest mismatch"
            );
            ensure!(
                path.state_indices.len() == path.token_ids.len() + 1,
                "canonical state sequence has the wrong length"
            );
            ensure!(
                path.state_indices.first() == Some(&self.root_state_index),
                "canonical path does not start at root"
            );
            ensure!(
                path.state_indices.last() == Some(&path.terminal_state_index),
                "canonical path does not end at its terminal"
            );
            ensure!(
                self.terminal_state_indices
                    .binary_search(&path.terminal_state_index)
                    .is_ok(),
                "canonical path terminal is not declared"
            );
            ensure!(
                path.token_ids.len() == CANONICAL_PATH_TOKEN_COUNT as usize,
                "canonical path has the wrong token count"
            );

            let mut reconstructed = Vec::new();
            for (step, &token) in path.token_ids.iter().enumerate() {
                let source = usize::try_from(path.state_indices[step])?;
                let expected_successor = path.state_indices[step + 1];
                let state = self
                    .states
                    .get(source)
                    .context("canonical source state is out of range")?;
                let start = usize::try_from(state.edge_start)?;
                let end = start + usize::try_from(state.edge_count)?;
                let edges = &self.edges[start..end];
                let edge_index = edges
                    .binary_search_by_key(&token, |edge| edge.token_id)
                    .map_err(|_| anyhow::anyhow!("canonical token is absent from state edges"))?;
                let edge = &edges[edge_index];
                ensure!(
                    edge.successor_state_index == expected_successor,
                    "canonical successor mismatch"
                );
                reconstructed.extend_from_slice(&decode_hex(&edge.piece_hex)?);
            }
            ensure!(
                reconstructed == target,
                "canonical target reconstruction failed"
            );
            let terminal = usize::try_from(path.terminal_state_index)?;
            ensure!(
                prefixes
                    .get(terminal)
                    .context("canonical terminal state is out of range")?
                    == &target,
                "canonical target differs from terminal prefix"
            );
            path_terminals.insert(path.terminal_state_index);
        }
        ensure!(
            path_terminals
                == self
                    .terminal_state_indices
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
            "canonical paths do not cover every terminal"
        );
        Ok(())
    }
}

pub(crate) fn derive_path_bounds(
    states: &[RuntimeState],
    edges: &[RuntimeEdge],
) -> Result<RuntimePathBounds> {
    ensure!(!states.is_empty(), "runtime state table is empty");
    let prefixes = states
        .iter()
        .map(|state| decode_hex(&state.prefix_hex))
        .collect::<Result<Vec<_>>>()?;
    let mut order: Vec<usize> = (0..states.len()).collect();
    order.sort_by_key(|&index| Reverse(prefixes[index].len()));
    let mut minimum = vec![None; states.len()];
    let mut maximum = vec![None; states.len()];

    for state_index in order {
        let state = &states[state_index];
        if state.kind == RuntimeStateKind::Terminal {
            minimum[state_index] = Some(0u32);
            maximum[state_index] = Some(0u32);
            continue;
        }
        let start = usize::try_from(state.edge_start)?;
        let end = start
            .checked_add(usize::try_from(state.edge_count)?)
            .context("runtime path-bound edge span overflow")?;
        ensure!(
            start < end && end <= edges.len(),
            "invalid path-bound edge span"
        );
        let mut state_minimum = u32::MAX;
        let mut state_maximum = 0u32;
        for edge in &edges[start..end] {
            let successor = usize::try_from(edge.successor_state_index)?;
            let successor_minimum = minimum
                .get(successor)
                .copied()
                .flatten()
                .context("successor minimum path bound is unavailable")?;
            let successor_maximum = maximum
                .get(successor)
                .copied()
                .flatten()
                .context("successor maximum path bound is unavailable")?;
            state_minimum = state_minimum.min(
                successor_minimum
                    .checked_add(1)
                    .context("minimum path bound overflow")?,
            );
            state_maximum = state_maximum.max(
                successor_maximum
                    .checked_add(1)
                    .context("maximum path bound overflow")?,
            );
        }
        minimum[state_index] = Some(state_minimum);
        maximum[state_index] = Some(state_maximum);
    }

    ensure!(
        minimum.iter().all(Option::is_some) && maximum.iter().all(Option::is_some),
        "not every runtime state reaches a terminal"
    );
    Ok(RuntimePathBounds {
        minimum_root_to_terminal_tokens: minimum[ROOT_STATE_INDEX as usize]
            .context("root minimum path bound is unavailable")?,
        maximum_root_to_terminal_tokens: maximum[ROOT_STATE_INDEX as usize]
            .context("root maximum path bound is unavailable")?,
    })
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_len(bytes: &mut Vec<u8>, length: usize) -> Result<()> {
    bytes.extend_from_slice(&u64::try_from(length)?.to_le_bytes());
    Ok(())
}

fn push_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<()> {
    push_len(output, value.len())?;
    output.extend_from_slice(value);
    Ok(())
}

fn decode_sha256(value: &str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "SHA-256 must be 64 lowercase hexadecimal characters"
    );
    let bytes = decode_hex(value)?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("SHA-256 decoded to the wrong length"))
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    ensure!(value.len() % 2 == 0, "hex string has odd length");
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "hex string must contain lowercase hexadecimal characters"
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

    const COMMITTED_RUNTIME: &[u8] =
        include_bytes!("../../../docs/bench/v0656-response-shape-runtime.json");

    #[test]
    fn committed_runtime_artifact_is_canonical_and_valid() {
        assert_eq!(
            format!("{:x}", Sha256::digest(COMMITTED_RUNTIME)),
            "69ca90ba2d4519052e4d453fd8e9759ca936d97131f5cb7ff786b2fc33cdeeb5"
        );
        let runtime: ResponseShapeRuntime =
            serde_json::from_slice(COMMITTED_RUNTIME).expect("parse committed runtime");
        runtime
            .validate_frozen()
            .expect("validate committed runtime");
        let mut canonical = serde_json::to_vec_pretty(&runtime).expect("serialize runtime");
        canonical.push(b'\n');
        assert_eq!(canonical, COMMITTED_RUNTIME);

        let original_digest = runtime.semantic_sha256().expect("semantic digest");
        let mut mutated = runtime;
        mutated.edges[0].token_id += 1;
        assert_ne!(
            mutated.semantic_sha256().expect("mutated semantic digest"),
            original_digest
        );
    }

    #[test]
    fn strict_hex_rejects_uppercase_and_odd_lengths() {
        assert_eq!(decode_hex("00ff").unwrap(), vec![0, 255]);
        assert!(decode_hex("0").is_err());
        assert!(decode_hex("AA").is_err());
        assert!(decode_sha256(&"0".repeat(64)).is_ok());
        assert!(decode_sha256(&"A".repeat(64)).is_err());
    }

    #[test]
    fn path_bounds_include_alternate_tokenizations() {
        let states = vec![
            RuntimeState {
                state_index: 0,
                kind: RuntimeStateKind::Branch,
                prefix_bytes: 0,
                prefix_hex: String::new(),
                bank_state_index: Some(0),
                edge_start: 0,
                edge_count: 2,
            },
            RuntimeState {
                state_index: 1,
                kind: RuntimeStateKind::Singleton,
                prefix_bytes: 1,
                prefix_hex: "61".into(),
                bank_state_index: Some(0),
                edge_start: 2,
                edge_count: 1,
            },
            RuntimeState {
                state_index: 2,
                kind: RuntimeStateKind::Terminal,
                prefix_bytes: 2,
                prefix_hex: "6162".into(),
                bank_state_index: None,
                edge_start: 3,
                edge_count: 0,
            },
        ];
        let edges = vec![
            RuntimeEdge {
                source_state_index: 0,
                token_id: 1,
                piece_bytes: 1,
                piece_hex: "61".into(),
                successor_state_index: 1,
            },
            RuntimeEdge {
                source_state_index: 0,
                token_id: 2,
                piece_bytes: 2,
                piece_hex: "6162".into(),
                successor_state_index: 2,
            },
            RuntimeEdge {
                source_state_index: 1,
                token_id: 3,
                piece_bytes: 1,
                piece_hex: "62".into(),
                successor_state_index: 2,
            },
        ];
        assert_eq!(
            derive_path_bounds(&states, &edges).unwrap(),
            RuntimePathBounds {
                minimum_root_to_terminal_tokens: 1,
                maximum_root_to_terminal_tokens: 2,
            }
        );
    }
}
