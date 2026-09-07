//! The family-neutral tail of every serve `generate`: turn a completed
//! `GenerationResult` into the wire outcome, the `serve stats:` line, and the
//! lossless terminal marker. Family-specific phase lines (`serve phases:`,
//! `serve dflash:`) are logged by the backend before calling this.

use super::events::{ServeStats, StopReason, Usage};
use super::http::GenerationOutcome;
use super::output_partition::GenerationEnd;
use crate::GenerationResult;

/// Map the decode loop's stop reason to the wire-facing pair. `GenerationEnd`
/// keeps the terminal token because Muse validates it against its grammar and
/// the partitioners need it; `StopReason` is the coarse label for logs.
pub(crate) fn generation_end(generation: &GenerationResult) -> (StopReason, GenerationEnd) {
    match generation.stop_reason {
        crate::StopReason::Eos => (
            StopReason::Eos,
            GenerationEnd::StopToken(
                *generation
                    .tokens
                    .last()
                    .expect("EOS generation includes its terminal token"),
            ),
        ),
        crate::StopReason::TokenLimit => (StopReason::TokenLimit, GenerationEnd::TokenLimit),
    }
}

pub(crate) fn decode_tps(generation: &GenerationResult) -> f64 {
    if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    }
}

/// Log the versioned `serve stats:` line and build the outcome. `matched_tokens`
/// and `restore_ms` are the prefix-restore facts (zero for lanes without a
/// cache); `prompt_tokens` is the rendered prompt length.
pub(crate) fn finish_generation(
    prompt_tokens: usize,
    generation: &GenerationResult,
    matched_tokens: usize,
    restore_ms: f64,
) -> GenerationOutcome {
    let (stop_reason, end) = generation_end(generation);
    tracing::info!(
        target: "qwen_diag",
        "serve stats: version=serve_stats_v1 prompt_tokens={} generated_tokens={} stop_reason={} matched_tokens={} restore_ms={:.1} decode_tps={:.2}",
        prompt_tokens,
        generation.tokens.len(),
        match stop_reason {
            StopReason::Eos => "eos",
            StopReason::TokenLimit => "token_limit",
        },
        matched_tokens,
        restore_ms,
        decode_tps(generation),
    );
    GenerationOutcome {
        end,
        usage: Usage {
            input_tokens: prompt_tokens,
            output_tokens: generation.tokens.len(),
            cached_tokens: matched_tokens,
        },
        stats: Some(ServeStats {
            matched_tokens,
            restore_ms,
            prompt_tokens,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation(
        tokens: Vec<i32>,
        stop_reason: crate::StopReason,
        wall_ms: f64,
    ) -> GenerationResult {
        GenerationResult {
            tokens,
            wall_ms,
            first_token_selection_ms: 0.0,
            first_token_ready_ms: None,
            first_token_callback_ms: None,
            transitions: 0,
            transition_ms: 0.0,
            first_transition_ms: None,
            stop_reason,
        }
    }

    #[test]
    fn eos_end_carries_the_terminal_token() {
        let g = generation(vec![5, 6, 151645], crate::StopReason::Eos, 10.0);
        assert_eq!(
            generation_end(&g),
            (StopReason::Eos, GenerationEnd::StopToken(151645))
        );
    }

    #[test]
    fn token_limit_end_has_no_terminal_token() {
        let g = generation(vec![5, 6], crate::StopReason::TokenLimit, 10.0);
        assert_eq!(
            generation_end(&g),
            (StopReason::TokenLimit, GenerationEnd::TokenLimit)
        );
    }

    #[test]
    fn outcome_counts_the_terminal_token_and_carries_restore_facts() {
        let g = generation(vec![5, 6, 7], crate::StopReason::Eos, 1_000.0);
        let outcome = finish_generation(19, &g, 12, 3.5);
        assert_eq!(outcome.usage.input_tokens, 19);
        assert_eq!(outcome.usage.output_tokens, 3);
        assert_eq!(outcome.usage.cached_tokens, 12);
        let stats = outcome.stats.unwrap();
        assert_eq!((stats.matched_tokens, stats.prompt_tokens), (12, 19));
        assert_eq!(stats.restore_ms, 3.5);
        assert_eq!(decode_tps(&g), 3.0);
        assert_eq!(
            decode_tps(&generation(vec![], crate::StopReason::TokenLimit, 0.0)),
            0.0
        );
    }
}
