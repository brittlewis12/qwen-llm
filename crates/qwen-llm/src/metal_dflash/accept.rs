//! CPU-only accept-prefix decisions; target state and emission stay with the caller.

use crate::sampling::{Sampler, SamplingError, SparseProposal, SpeculativeSamplingDecision};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashNextCarry {
    Correction(i32),
    Bonus(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DFlashAcceptStopReason {
    Eos,
    TokenLimit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DFlashAcceptDecision {
    /// Newly accepted output tokens, including a terminal token if present.
    /// The already-emitted anchor is never repeated here.
    pub accepted_drafts: Vec<i32>,
    /// Not yet emitted or consumed. The caller checks EOS/budget when emitting
    /// this token, before attempting another target transition.
    pub next_carry: Option<DFlashNextCarry>,
    /// Terminal accepted draft, terminal anchor, or exhausted output budget.
    /// EOS takes precedence when EOS and the output cap coincide.
    pub stop_reason: Option<DFlashAcceptStopReason>,
    /// Retain exactly `verified_tokens[..retain_input_count]` and its captures.
    /// Normally anchor + accepted drafts; exclude the final terminal token.
    /// Zero means restore the pre-block state, leaving the anchor unconsumed.
    pub retain_input_count: usize,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum DFlashAcceptError {
    #[error("invalid DFlash accept-prefix input: {0}")]
    InvalidInput(&'static str),
    #[error("verified input token {token} at index {index} is outside vocabulary {vocab}")]
    InvalidToken {
        index: usize,
        token: i32,
        vocab: usize,
    },
    #[error("proposal row {row} sampled token {proposal}, but verified draft is {draft}")]
    ProposalMismatch {
        row: usize,
        proposal: i32,
        draft: i32,
    },
    #[error("DFlash target row {row}: {source}")]
    Sampling { row: usize, source: SamplingError },
}

/// Decide one verified block without mutating target state or emitting tokens.
///
/// `verified_tokens` is `[already_emitted_unconsumed_carry, draft...]` (nonempty).
/// `target_logits` must contain exactly `verified_tokens.len() * vocab` elements:
/// row `r` predicts input `r + 1`, and the last row supplies the all-accepted
/// bonus (also for an anchor-only block). Sparse proposal row `r` must describe
/// the actual distribution that sampled input `r + 1`; omit proposals only for
/// deterministic drafts. Target filtering/coupling is entirely owned by Sampler.
///
/// `remaining_output_budget` excludes the already-emitted carry. At zero budget
/// (or a stop-ID carry), return no new output, no draws and zero retained inputs.
/// Accepted EOS/cap ends sampling immediately and leaves that final accepted
/// token unconsumed: `retain_input_count == accepted_drafts.len()`. Otherwise
/// retain anchor + accepted drafts; correction/bonus is a new unconsumed carry,
/// not part of `accepted_drafts`. The caller must emit it and check EOS/cap.
///
/// Extents, verified IDs and proposal alignment/IDs are checked for the whole
/// block before sampling, even for unused rows. Logit values and proposal weights
/// are validated by Sampler only on reached rows; unused tails are not sampled.
/// Errors leave the request sampler unchanged, including errors after earlier
/// accepted rows. Callers own checkpoint restore/replay, retained capture append,
/// and callback emission (including suppressing stop-ID callbacks).
pub fn decide_dflash_accept_prefix(
    verified_tokens: &[i32],
    target_logits: &[f32],
    vocab: usize,
    proposals: Option<&[SparseProposal]>,
    sampler: &mut Sampler,
    remaining_output_budget: usize,
    stop_ids: &[i32],
) -> Result<DFlashAcceptDecision, DFlashAcceptError> {
    if verified_tokens.is_empty() {
        return Err(DFlashAcceptError::InvalidInput("missing carry token"));
    }
    if vocab == 0 || vocab > i32::MAX as usize {
        return Err(DFlashAcceptError::InvalidInput("invalid vocabulary size"));
    }
    if verified_tokens.len().checked_mul(vocab) != Some(target_logits.len()) {
        return Err(DFlashAcceptError::InvalidInput(
            "target logits extent mismatch",
        ));
    }
    for (index, &token) in verified_tokens.iter().enumerate() {
        if token < 0 || token as usize >= vocab {
            return Err(DFlashAcceptError::InvalidToken {
                index,
                token,
                vocab,
            });
        }
    }
    let n_drafts = verified_tokens.len() - 1;
    if let Some(proposals) = proposals {
        if proposals.len() != n_drafts {
            return Err(DFlashAcceptError::InvalidInput(
                "proposal row count mismatch",
            ));
        }
        for (row, proposal) in proposals.iter().enumerate() {
            let draft = verified_tokens[row + 1];
            if proposal.token != draft {
                return Err(DFlashAcceptError::ProposalMismatch {
                    row,
                    proposal: proposal.token,
                    draft,
                });
            }
            for candidate in &proposal.candidates {
                if candidate.token < 0 || candidate.token as usize >= vocab {
                    return Err(DFlashAcceptError::Sampling {
                        row,
                        source: SamplingError::ProposalTokenOutOfRange(candidate.token),
                    });
                }
            }
        }
    }
    let anchor_stop = if stop_ids.contains(&verified_tokens[0]) {
        Some(DFlashAcceptStopReason::Eos)
    } else if remaining_output_budget == 0 {
        Some(DFlashAcceptStopReason::TokenLimit)
    } else {
        None
    };
    if anchor_stop.is_some() {
        return Ok(DFlashAcceptDecision {
            accepted_drafts: Vec::new(),
            next_carry: None,
            stop_reason: anchor_stop,
            retain_input_count: 0,
        });
    }

    let mut next_sampler = sampler.clone();
    let mut accepted_drafts = Vec::with_capacity(n_drafts.min(remaining_output_budget));
    let (next_carry, stop_reason) = loop {
        let row = accepted_drafts.len();
        let logits = &target_logits[row * vocab..(row + 1) * vocab];
        let sampling_error = |source| DFlashAcceptError::Sampling { row, source };
        if row == n_drafts {
            let bonus = next_sampler.sample(logits).map_err(sampling_error)?.token;
            break (Some(DFlashNextCarry::Bonus(bonus)), None);
        }
        let draft = verified_tokens[row + 1];
        let decision = if let Some(proposals) = proposals {
            next_sampler.couple_sparse_proposal(logits, &proposals[row])
        } else {
            next_sampler.couple_deterministic_proposal(logits, draft)
        }
        .map_err(sampling_error)?;
        if let SpeculativeSamplingDecision::Rejected { correction } = decision {
            break (Some(DFlashNextCarry::Correction(correction)), None);
        }
        accepted_drafts.push(draft);
        if stop_ids.contains(&draft) {
            break (None, Some(DFlashAcceptStopReason::Eos));
        }
        if accepted_drafts.len() == remaining_output_budget {
            break (None, Some(DFlashAcceptStopReason::TokenLimit));
        }
    };
    let retain_input_count = 1 + accepted_drafts.len() - usize::from(stop_reason.is_some());
    *sampler = next_sampler;
    Ok(DFlashAcceptDecision {
        accepted_drafts,
        next_carry,
        stop_reason,
        retain_input_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{SamplingConfig, WeightedCandidate};

    const VOCAB: usize = 6;
    const INPUT: [i32; 4] = [0, 1, 2, 3];

    fn sampler(temperature: f32) -> Sampler {
        Sampler::new(SamplingConfig {
            temperature,
            top_k: 1,
            seed: 17,
            ..SamplingConfig::default()
        })
        .unwrap()
    }

    fn logits(winners: &[i32]) -> Vec<f32> {
        let mut values = vec![-10.0; winners.len() * VOCAB];
        for (row, &token) in winners.iter().enumerate() {
            values[row * VOCAB + token as usize] = 10.0;
        }
        values
    }

    fn proposal(token: i32, candidates: &[(i32, f64)]) -> SparseProposal {
        SparseProposal {
            token,
            candidates: candidates
                .iter()
                .map(|&(token, weight)| WeightedCandidate { token, weight })
                .collect(),
        }
    }

    fn one_hot_proposals() -> Vec<SparseProposal> {
        INPUT[1..]
            .iter()
            .map(|&t| proposal(t, &[(t, 1.0)]))
            .collect()
    }

    fn assert_same_rng(actual: &Sampler, expected: &Sampler) {
        assert_eq!(actual.draws(), expected.draws());
        assert_eq!(
            actual.diagnose_next(&[0.0; VOCAB]),
            expected.diagnose_next(&[0.0; VOCAB]),
        );
    }

    #[test]
    fn every_rejection_position_keeps_anchor_and_only_accepted_inputs() {
        for temperature in [0.0, 1.0] {
            for sparse in [false, true] {
                for rejected_row in 0..3 {
                    let mut winners = [1, 2, 3, 4];
                    winners[rejected_row] = 5;
                    let mut rows = logits(&winners);
                    rows[(rejected_row + 1) * VOCAB..].fill(f32::NAN);
                    let proposals = one_hot_proposals();
                    let mut request = sampler(temperature);
                    let decision = decide_dflash_accept_prefix(
                        &INPUT,
                        &rows,
                        VOCAB,
                        sparse.then_some(proposals.as_slice()),
                        &mut request,
                        4,
                        &[],
                    )
                    .unwrap();
                    assert_eq!(decision.accepted_drafts, INPUT[1..1 + rejected_row]);
                    assert_eq!(decision.next_carry, Some(DFlashNextCarry::Correction(5)));
                    assert_eq!(decision.stop_reason, None);
                    assert_eq!(decision.retain_input_count, 1 + rejected_row);
                    let draws = if sparse {
                        rejected_row + 2
                    } else if temperature > 0.0 {
                        rejected_row + 1
                    } else {
                        0
                    };
                    let expected = Sampler::at_draw(request.config(), draws).unwrap();
                    assert_same_rng(&request, &expected);
                }
            }
        }
    }

    #[test]
    fn all_accepted_samples_bonus_from_last_row() {
        for temperature in [0.0, 1.0] {
            for sparse in [false, true] {
                let mut request = sampler(temperature);
                let proposals = one_hot_proposals();
                let decision = decide_dflash_accept_prefix(
                    &INPUT,
                    &logits(&[1, 2, 3, 4]),
                    VOCAB,
                    sparse.then_some(proposals.as_slice()),
                    &mut request,
                    4,
                    &[],
                )
                .unwrap();
                assert_eq!(
                    decision,
                    DFlashAcceptDecision {
                        accepted_drafts: vec![1, 2, 3],
                        next_carry: Some(DFlashNextCarry::Bonus(4)),
                        stop_reason: None,
                        retain_input_count: 4,
                    }
                );
                let draws = if temperature > 0.0 {
                    4
                } else if sparse {
                    3
                } else {
                    0
                };
                assert_same_rng(
                    &request,
                    &Sampler::at_draw(request.config(), draws).unwrap(),
                );
            }
        }
    }

    #[test]
    fn eos_and_cap_at_every_accepted_position_leave_terminal_unconsumed() {
        for sparse in [false, true] {
            for terminal_count in 1..=3 {
                // EOS alone, cap alone, and EOS taking precedence over the cap.
                for (eos, cap) in [(true, false), (false, true), (true, true)] {
                    let mut rows = logits(&[1, 2, 3, 4]);
                    rows[terminal_count * VOCAB..].fill(f32::NAN);
                    let mut proposals = one_hot_proposals();
                    for tail in &mut proposals[terminal_count..] {
                        tail.candidates[0].weight = f64::NAN;
                    }
                    let stops = if eos {
                        vec![INPUT[terminal_count]]
                    } else {
                        vec![]
                    };
                    let mut request = sampler(1.0);
                    let decision = decide_dflash_accept_prefix(
                        &INPUT,
                        &rows,
                        VOCAB,
                        sparse.then_some(proposals.as_slice()),
                        &mut request,
                        if cap { terminal_count } else { 4 },
                        &stops,
                    )
                    .unwrap();
                    assert_eq!(decision.accepted_drafts, INPUT[1..=terminal_count]);
                    assert_eq!(decision.next_carry, None);
                    assert_eq!(
                        decision.stop_reason,
                        Some(if eos {
                            DFlashAcceptStopReason::Eos
                        } else {
                            DFlashAcceptStopReason::TokenLimit
                        })
                    );
                    assert_eq!(decision.retain_input_count, terminal_count);
                    assert_same_rng(
                        &request,
                        &Sampler::at_draw(request.config(), terminal_count).unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn terminal_anchor_and_zero_budget_do_not_sample_or_retain_any_input() {
        for input in [&INPUT[..1], &INPUT[..]] {
            for (budget, stops, reason) in [
                (0, &[][..], DFlashAcceptStopReason::TokenLimit),
                (0, &[0][..], DFlashAcceptStopReason::Eos),
                (4, &[0][..], DFlashAcceptStopReason::Eos),
            ] {
                let mut request = sampler(1.0);
                let before = request.clone();
                let decision = decide_dflash_accept_prefix(
                    input,
                    &vec![f32::NAN; input.len() * VOCAB],
                    VOCAB,
                    None,
                    &mut request,
                    budget,
                    stops,
                )
                .unwrap();
                assert_eq!(
                    decision,
                    DFlashAcceptDecision {
                        accepted_drafts: vec![],
                        next_carry: None,
                        stop_reason: Some(reason),
                        retain_input_count: 0,
                    }
                );
                assert_same_rng(&request, &before);
            }
        }
    }

    #[test]
    fn anchor_only_bonus_and_rejection_carry_are_not_emitted_or_consumed() {
        for proposals in [None, Some(&[][..])] {
            let mut request = sampler(1.0);
            let decision = decide_dflash_accept_prefix(
                &[0],
                &logits(&[4]),
                VOCAB,
                proposals,
                &mut request,
                1,
                &[4],
            )
            .unwrap();
            assert_eq!(
                decision,
                DFlashAcceptDecision {
                    accepted_drafts: vec![],
                    next_carry: Some(DFlashNextCarry::Bonus(4)),
                    stop_reason: None,
                    retain_input_count: 1,
                }
            );
            assert_eq!(request.draws(), 1);
        }
        let decision = decide_dflash_accept_prefix(
            &INPUT,
            &logits(&[5, 2, 3, 4]),
            VOCAB,
            None,
            &mut sampler(1.0),
            1,
            &[5],
        )
        .unwrap();
        assert_eq!(
            decision,
            DFlashAcceptDecision {
                accepted_drafts: vec![],
                next_carry: Some(DFlashNextCarry::Correction(5)),
                stop_reason: None,
                retain_input_count: 1,
            }
        );
    }

    #[test]
    fn sparse_coupling_uses_actual_q_and_the_request_target_filters() {
        let target_row = [2.0, 1.9, 1.5, 0.5, -1.0, -2.0];
        let mut rows = target_row.to_vec();
        rows.extend(logits(&[4]));
        let high_q = proposal(0, &[(0, 8.0), (1, 1.0), (5, 1.0)]);
        let low_q = proposal(0, &[(0, 1.0), (1, 99.0)]);
        let mut accepted = 0;
        let mut rejected = 0;
        for seed in 0..64 {
            let config = SamplingConfig {
                temperature: 0.7,
                top_k: 4,
                top_p: 0.7,
                min_p: 0.5,
                seed,
            };
            let mut request = Sampler::new(config).unwrap();
            let mut reference = request.clone();
            let expected = match reference
                .couple_sparse_proposal(&target_row, &high_q)
                .unwrap()
            {
                SpeculativeSamplingDecision::Accepted => {
                    accepted += 1;
                    DFlashAcceptDecision {
                        accepted_drafts: vec![0],
                        next_carry: Some(DFlashNextCarry::Bonus(
                            reference.sample(&rows[VOCAB..]).unwrap().token,
                        )),
                        stop_reason: None,
                        retain_input_count: 2,
                    }
                }
                SpeculativeSamplingDecision::Rejected { correction } => {
                    rejected += 1;
                    assert_eq!(correction, 1);
                    DFlashAcceptDecision {
                        accepted_drafts: vec![],
                        next_carry: Some(DFlashNextCarry::Correction(correction)),
                        stop_reason: None,
                        retain_input_count: 1,
                    }
                }
            };
            let actual = decide_dflash_accept_prefix(
                &[3, 0],
                &rows,
                VOCAB,
                Some(std::slice::from_ref(&high_q)),
                &mut request,
                2,
                &[],
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_same_rng(&request, &reference);

            let low = decide_dflash_accept_prefix(
                &[3, 0],
                &rows,
                VOCAB,
                Some(std::slice::from_ref(&low_q)),
                &mut Sampler::new(config).unwrap(),
                2,
                &[],
            )
            .unwrap();
            assert_eq!(low.accepted_drafts, [0]);

            // Token 2 is removed by target top-p after top-k/min-p/temperature.
            // It would be accepted with this small q if target filters were lost.
            let filtered = proposal(2, &[(2, 1.0), (0, 99.0)]);
            let mut request = Sampler::new(config).unwrap();
            let filtered = decide_dflash_accept_prefix(
                &[3, 2],
                &rows,
                VOCAB,
                Some(&[filtered]),
                &mut request,
                2,
                &[],
            )
            .unwrap();
            assert!(filtered.accepted_drafts.is_empty());
            assert!(matches!(
                filtered.next_carry,
                Some(DFlashNextCarry::Correction(0 | 1))
            ));
            assert_eq!(request.draws(), 2);
        }
        assert!(accepted > 0 && rejected > 0);
    }

    #[test]
    fn malformed_shapes_tokens_and_late_proposal_alignment_leave_rng_unchanged() {
        let rows = logits(&[1, 2, 3, 4]);
        let proposals = one_hot_proposals();
        let mut mismatch = proposals.clone();
        mismatch[2].token = 4;
        let mut invalid_candidate = proposals.clone();
        invalid_candidate[2].candidates[0].token = -1;
        let invalid_inputs: &[(&[i32], &[f32], usize, Option<&[SparseProposal]>)] = &[
            (&[], &[], VOCAB, None),
            (&INPUT, &rows, 0, None),
            (&INPUT, &rows, usize::MAX, None),
            (&INPUT, &rows[..rows.len() - 1], VOCAB, None),
            (&INPUT[..3], &rows, VOCAB, None),
            (&[-1, 1, 2, 3], &rows, VOCAB, None),
            (&[0, 1, 2, 6], &rows, VOCAB, None),
            (&INPUT, &rows, VOCAB, Some(&proposals[..2])),
            (&INPUT, &rows, VOCAB, Some(&mismatch)),
            (&INPUT, &rows, VOCAB, Some(&invalid_candidate)),
        ];
        for &(input, rows, vocab, proposals) in invalid_inputs {
            for budget in [0, 4] {
                let mut request = sampler(1.0);
                let before = request.clone();
                assert!(
                    decide_dflash_accept_prefix(
                        input,
                        rows,
                        vocab,
                        proposals,
                        &mut request,
                        budget,
                        &[],
                    )
                    .is_err()
                );
                assert_same_rng(&request, &before);
            }
        }
    }

    #[test]
    fn errors_after_accepted_rows_roll_back_the_request_rng() {
        let mut rows = logits(&[1, 2, 3, 4]);
        for row in 1..4 {
            rows[row * VOCAB] = f32::NAN;
            let mut request = sampler(1.0);
            let before = request.clone();
            assert!(matches!(decide_dflash_accept_prefix(
                &INPUT, &rows, VOCAB, None, &mut request, 4, &[],
            ), Err(DFlashAcceptError::Sampling { row: failed, .. }) if failed == row));
            assert_same_rng(&request, &before);
            rows = logits(&[1, 2, 3, 4]);
        }
        for invalid in [
            proposal(2, &[]),
            proposal(2, &[(2, -1.0)]),
            proposal(2, &[(2, f64::NAN)]),
            proposal(2, &[(2, 0.0)]),
            proposal(2, &[(1, 1.0)]),
        ] {
            let mut proposals = one_hot_proposals();
            proposals[1] = invalid;
            let mut request = sampler(1.0);
            let before = request.clone();
            assert!(matches!(
                decide_dflash_accept_prefix(
                    &INPUT,
                    &rows,
                    VOCAB,
                    Some(&proposals),
                    &mut request,
                    4,
                    &[],
                ),
                Err(DFlashAcceptError::Sampling { row: 1, .. })
            ));
            assert_same_rng(&request, &before);
        }
    }
}
