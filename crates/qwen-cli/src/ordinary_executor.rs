//! Shared serial token lifecycle. Callers own sampling, forwards, capture,
//! transport and process shutdown; this module owns their ordering, not policy.

use anyhow::{Result, ensure};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Route diagnostic work through the original token forward. Callers retain
/// checkpoints, sequence validation/advancement and post-consumption observation.
pub(crate) fn post_block_forward(
    forward: &qwen_llm::metal_forward::MetalForward<'_>,
    session: &mut qwen_llm::metal_forward::MetalSession,
    token: i32,
    position: u32,
    needs_logits: bool,
    capture: Option<(&[u32], &qwen_llm::metal::MetalTensor)>,
    operations: &[qwen_llm::metal::PostBlockIntervention<'_>],
) -> Result<Vec<f32>> {
    match (capture, needs_logits) {
        (Some((layers, capture)), true) => Ok(forward.single_token_with_post_block_interventions(
            token, position, session, layers, capture, operations,
        )?),
        (Some((layers, capture)), false) => {
            forward.single_token_with_post_block_interventions_no_tail(
                token, position, session, layers, capture, operations,
            )?;
            Ok(Vec::new())
        }
        (None, true) => Ok(
            forward.single_token_with_post_block_interventions_no_capture(
                token, position, session, operations,
            )?,
        ),
        (None, false) => {
            forward.single_token_with_post_block_interventions_no_capture_no_tail(
                token, position, session, operations,
            )?;
            Ok(Vec::new())
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct ExecutionControl {
    cancelled: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutionCancelled;

impl std::fmt::Display for ExecutionCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("generation cancelled")
    }
}
impl std::error::Error for ExecutionCancelled {}

impl ExecutionControl {
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Explicit request cancellation only. Compose process checkpoints at the
    /// caller so signal diagnostics and shutdown ownership remain intact.
    pub(crate) fn checkpoint(&self) -> Result<(), ExecutionCancelled> {
        if self.is_cancelled() {
            Err(ExecutionCancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum TokenAllocation {
    Upfront,
    #[allow(dead_code)] // Constructed by the separate qwen-lens binary.
    Incremental,
}

pub(crate) struct DecodeOptions<'a> {
    pub(crate) max_tokens: usize,
    pub(crate) stop_tokens: &'a [i32],
    pub(crate) allocation: TokenAllocation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalReason {
    StopToken,
    TokenLimit,
}

#[derive(Debug)]
pub(crate) struct DecodeSummary {
    pub(crate) tokens: Vec<i32>,
    pub(crate) wall_ms: f64,
    pub(crate) first_token_selection_ms: f64,
    pub(crate) first_token_ready_ms: Option<f64>,
    pub(crate) first_token_callback_ms: Option<f64>,
    pub(crate) transitions: usize,
    pub(crate) transition_ms: f64,
    pub(crate) first_transition_ms: Option<f64>,
    pub(crate) stop_reason: TerminalReason,
}

/// Stop tokens are retained but neither published nor consumed. A non-stop
/// sample at the token limit is published, then retained without a forward.
pub(crate) fn decode<Context, State>(
    mut state: State,
    options: DecodeOptions<'_>,
    context: &mut Context,
    mut checkpoint: impl FnMut() -> Result<()>,
    mut select: impl FnMut(&mut Context, &State) -> Result<i32>,
    mut on_token: impl FnMut(i32) -> Result<()>,
    mut transition: impl FnMut(&mut Context, i32) -> Result<State>,
) -> Result<DecodeSummary> {
    ensure!(options.max_tokens > 0, "max_tokens must be >= 1");
    let wall_t0 = Instant::now();
    let mut tokens = match options.allocation {
        TokenAllocation::Upfront => Vec::with_capacity(options.max_tokens),
        TokenAllocation::Incremental => Vec::new(),
    };
    let mut first_token_selection_ms = None;
    let mut first_token_ready_ms = None;
    let mut first_token_callback_ms = None;
    let mut transitions = 0;
    let mut transition_ms = 0.0;
    let mut first_transition_ms = None;
    let stop_reason = loop {
        checkpoint()?;
        let selection_t0 = Instant::now();
        let token = select(context, &state)?;
        first_token_selection_ms.get_or_insert_with(|| selection_t0.elapsed().as_secs_f64() * 1e3);
        first_token_ready_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        tokens.push(token);
        if options.stop_tokens.contains(&token) {
            break TerminalReason::StopToken;
        }
        on_token(token)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if tokens.len() == options.max_tokens {
            break TerminalReason::TokenLimit;
        }
        // Publication can request cancellation. Do not consume that token just
        // to discover the cancellation after an otherwise avoidable forward.
        checkpoint()?;
        let transition_t0 = Instant::now();
        state = transition(context, token)?;
        checkpoint()?;
        let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
        transition_ms += elapsed_ms;
        first_transition_ms.get_or_insert(elapsed_ms);
        transitions += 1;
    };
    Ok(DecodeSummary {
        tokens,
        wall_ms: wall_t0.elapsed().as_secs_f64() * 1e3,
        first_token_selection_ms: first_token_selection_ms.unwrap_or(0.0),
        first_token_ready_ms,
        first_token_callback_ms,
        transitions,
        transition_ms,
        first_transition_ms,
        stop_reason,
    })
}

#[cfg(test)]
mod tests;
