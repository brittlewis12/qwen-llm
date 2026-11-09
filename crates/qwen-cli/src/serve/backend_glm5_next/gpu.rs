//! Shared serial/bounded-owner adapter for GLM GPU feedback.
use super::*;
use crate::serve::decode_cursor::{Advance, DecodeCursor};
pub(super) fn advance(
    ctx: &qwen_llm::metal::MetalContext,
    tokenizer: &qwen_llm::tokenizer::NativeTokenizer,
    session: &mut Glm5NextSession<'_>,
    sampler: &mut qwen_llm::sampling::Sampler,
    cursor: &mut DecodeCursor,
    sink: &mut dyn GenerationSink,
    limit: usize,
) -> Option<Result<Advance, BackendFailure>> {
    if !qwen_llm::glm5_next_metal::gpu_token_loop_enabled(ctx) || cursor.remaining_tokens() < 32 {
        return None;
    }
    let logits = cursor.available_logits()?;
    // A newly arriving peer waits at most one bounded burst. Existing
    // peers/prefills use native rounds, and backpressure yields at a token.
    let count = cursor.remaining_tokens().min(limit);
    let offset = cursor.elapsed_ms();
    let draws = sampler.draws();
    let mut tokens = Vec::with_capacity(count);
    let mut callback_error = None;
    let result = session.generate_gpu_tokens(ctx, logits, sampler, count, &CHAT_STOPS, |token| {
        tokens.push(token);
        let result = (|| -> Result<bool, BackendFailure> {
            crate::shutdown::checkpoint().map_err(|e| ServeError::server_error(e.to_string()))?;
            let bytes = tokenizer
                .try_decode_piece_bytes_exact(token)
                .map_err(|e| ServeError::server_error(format!("decode {FAMILY}: {e}")))?;
            sink.piece(bytes)
                .and_then(|()| sink.tick())
                .map_err(BackendFailure::Aborted)?;
            let ready = sink.flush_ready().map_err(BackendFailure::Aborted)?;
            crate::shutdown::checkpoint().map_err(|e| ServeError::server_error(e.to_string()))?;
            Ok(ready)
        })();
        match result {
            Ok(ready) => ready,
            Err(e) => {
                callback_error = Some(e);
                false
            }
        }
    });
    if let Some(e) = callback_error {
        return Some(Err(e));
    }
    match result {
        Err(_) if !session.is_poisoned() && tokens.is_empty() && sampler.draws() == draws => None,
        Err(e) => Some(Err(ServeError::server_error(format!(
            "{FAMILY} GPU feedback: {e}"
        ))
        .into())),
        Ok(stats) => {
            if stats.stopped_on_eos {
                tokens.push(stats.terminal_token.expect("GPU EOS token"));
            }
            Some(Ok(cursor.append_gpu_burst(&tokens, stats, offset)))
        }
    }
}
