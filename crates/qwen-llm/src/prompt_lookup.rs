//! Exact prompt and committed-output lookup for speculative proposals.

use rustc_hash::FxHashMap;

pub const MATCH_TOKENS: usize = 8;
pub const DRAFT_TOKENS: usize = 7;

type MatchKey = [i32; MATCH_TOKENS];
type RecentIndex = FxHashMap<MatchKey, usize>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProposalSource {
    Prompt,
    SelfOutput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptLookupCandidate {
    pub source: ProposalSource,
    pub source_start: usize,
    pub source_end: usize,
    pub absolute_source_end: usize,
    pub match_len: usize,
    pub proposal: [i32; DRAFT_TOKENS],
}

pub struct PromptLookupProposer {
    prompt: Box<[i32]>,
    prompt_index: RecentIndex,
    committed: Vec<i32>,
    self_index: RecentIndex,
    next_self_end: usize,
}

impl PromptLookupProposer {
    pub fn new(prompt: &[i32]) -> Self {
        Self {
            prompt: prompt.into(),
            prompt_index: build_recent_index(prompt),
            committed: Vec::new(),
            self_index: RecentIndex::default(),
            next_self_end: MATCH_TOKENS,
        }
    }

    /// Add generated tokens only after target verification and rollback complete.
    pub fn commit_verified(&mut self, tokens: &[i32]) {
        self.committed.extend_from_slice(tokens);
        while self.next_self_end + DRAFT_TOKENS <= self.committed.len() {
            let end = self.next_self_end;
            let key = key_ending_at(&self.committed, end);
            self.self_index.insert(key, end);
            self.next_self_end += 1;
        }
    }

    pub fn propose(&self) -> Option<PromptLookupCandidate> {
        let key = visible_suffix_key(&self.prompt, &self.committed)?;
        let prompt_end = self.prompt_index.get(&key).copied();
        let self_end = self.self_index.get(&key).copied();
        match (prompt_end, self_end) {
            (Some(prompt_end), Some(self_end)) => {
                let self_absolute_end = self.prompt.len() + self_end;
                Some(if prompt_end > self_absolute_end {
                    candidate(ProposalSource::Prompt, &self.prompt, prompt_end, 0)
                } else {
                    candidate(
                        ProposalSource::SelfOutput,
                        &self.committed,
                        self_end,
                        self.prompt.len(),
                    )
                })
            }
            (Some(end), None) => Some(candidate(ProposalSource::Prompt, &self.prompt, end, 0)),
            (None, Some(end)) => Some(candidate(
                ProposalSource::SelfOutput,
                &self.committed,
                end,
                self.prompt.len(),
            )),
            (None, None) => None,
        }
    }

    pub fn committed_len(&self) -> usize {
        self.committed.len()
    }
}

fn build_recent_index(tokens: &[i32]) -> RecentIndex {
    let mut index = RecentIndex::default();
    let Some(last_end) = tokens.len().checked_sub(DRAFT_TOKENS) else {
        return index;
    };
    for end in MATCH_TOKENS..=last_end {
        index.insert(key_ending_at(tokens, end), end);
    }
    index
}

fn key_ending_at(tokens: &[i32], end: usize) -> MatchKey {
    tokens[end - MATCH_TOKENS..end]
        .try_into()
        .expect("fixed prompt-lookup match width")
}

fn visible_suffix_key(prompt: &[i32], committed: &[i32]) -> Option<MatchKey> {
    let total = prompt.len() + committed.len();
    let start = total.checked_sub(MATCH_TOKENS)?;
    let mut key = [0; MATCH_TOKENS];
    for (slot, absolute) in key.iter_mut().zip(start..total) {
        *slot = if absolute < prompt.len() {
            prompt[absolute]
        } else {
            committed[absolute - prompt.len()]
        };
    }
    Some(key)
}

fn candidate(
    source_kind: ProposalSource,
    source: &[i32],
    source_end: usize,
    absolute_offset: usize,
) -> PromptLookupCandidate {
    PromptLookupCandidate {
        source: source_kind,
        source_start: source_end - MATCH_TOKENS,
        source_end,
        absolute_source_end: absolute_offset + source_end,
        match_len: MATCH_TOKENS,
        proposal: source[source_end..source_end + DRAFT_TOKENS]
            .try_into()
            .expect("indexed source has a full proposal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_index_boundaries_are_inclusive() {
        let short = PromptLookupProposer::new(&[1; MATCH_TOKENS + DRAFT_TOKENS - 1]);
        assert!(short.prompt_index.is_empty());

        let exact = PromptLookupProposer::new(&[1; MATCH_TOKENS + DRAFT_TOKENS]);
        assert_eq!(exact.prompt_index.len(), 1);
        assert_eq!(
            exact.prompt_index.values().copied().next(),
            Some(MATCH_TOKENS)
        );
    }

    #[test]
    fn self_index_waits_for_full_committed_continuation() {
        let mut proposer = PromptLookupProposer::new(&[]);
        proposer.commit_verified(&[1; MATCH_TOKENS + DRAFT_TOKENS - 1]);
        assert!(proposer.self_index.is_empty());
        proposer.commit_verified(&[1]);
        assert_eq!(
            proposer.self_index.values().copied().next(),
            Some(MATCH_TOKENS)
        );
    }

    #[test]
    fn prompt_copy_matches_offline_fixture() {
        let prompt = [
            1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7,
        ];
        let mut proposer = PromptLookupProposer::new(&prompt);
        proposer.commit_verified(&[8]);
        let candidate = proposer.propose().expect("prompt proposal");
        assert_eq!(candidate.source, ProposalSource::Prompt);
        assert_eq!(candidate.source_end, 8);
        assert_eq!(candidate.proposal, [11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn most_recent_prompt_occurrence_wins() {
        let prompt = [
            1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 22,
            23, 24, 25, 26, 27, 0, 1, 2, 3, 4, 5, 6, 7,
        ];
        let mut proposer = PromptLookupProposer::new(&prompt);
        proposer.commit_verified(&[8]);
        let candidate = proposer.propose().expect("recent prompt proposal");
        assert_eq!(candidate.source_end, 24);
        assert_eq!(candidate.proposal, [21, 22, 23, 24, 25, 26, 27]);
    }

    #[test]
    fn eligible_self_occurrence_beats_prompt() {
        let prompt = [
            1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7,
        ];
        let mut proposer = PromptLookupProposer::new(&prompt);
        proposer.commit_verified(&[
            1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33, 34, 35, 36, 37, 1, 2, 3, 4, 5, 6, 7, 8,
        ]);
        let candidate = proposer.propose().expect("self proposal");
        assert_eq!(candidate.source, ProposalSource::SelfOutput);
        assert_eq!(candidate.proposal, [31, 32, 33, 34, 35, 36, 37]);
    }

    #[test]
    fn batched_commit_indexes_every_new_endpoint() {
        let mut proposer = PromptLookupProposer::new(&[]);
        proposer.commit_verified(&(0..32).collect::<Vec<_>>());
        assert_eq!(proposer.next_self_end, 26);
        assert_eq!(proposer.self_index.len(), 18);
    }

    #[test]
    fn recent_selector_ignores_older_longer_match() {
        let prompt = [
            9, 9, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8,
            21, 22, 23, 24, 25, 26, 27, 0, 1, 2, 3, 4, 5, 6, 7,
        ];
        let mut proposer = PromptLookupProposer::new(&prompt);
        proposer.commit_verified(&[8]);
        let candidate = proposer.propose().expect("recent proposal");
        assert_eq!(candidate.source_end, 26);
        assert_eq!(candidate.proposal, [21, 22, 23, 24, 25, 26, 27]);
    }

    #[test]
    fn repetitive_prompt_keeps_one_recent_endpoint() {
        let proposer = PromptLookupProposer::new(&[1; 32_768]);
        assert_eq!(proposer.prompt_index.len(), 1);
        assert_eq!(
            proposer.prompt_index.values().copied().next(),
            Some(32_768 - DRAFT_TOKENS)
        );
    }
}
