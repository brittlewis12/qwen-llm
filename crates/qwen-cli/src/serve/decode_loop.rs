//! The serial decode loop every resident backend runs after its own prefill.
//!
//! Serial only: one token per forward. Verifier-driven multi-token steps
//! (DFlash in `backend.rs`) do not fit `forward` and are not meant to.

use super::http::{BackendFailure, GenerationSink};
use super::items::ServeError;
use anyhow::Context as _;
use qwen_llm::sampling::Sampler;
use qwen_llm::tokenizer::{TokError, Tokenize};
use std::io;

/// Request-scoped inputs to the loop; everything a backend derives from its
/// prefill plus the request.
pub(crate) struct DecodeRequest<'a> {
    pub(crate) family: &'static str,
    pub(crate) logits: Vec<f32>,
    pub(crate) max_tokens: usize,
    pub(crate) stop_tokens: &'a [i32],
    pub(crate) vocab_size: u32,
}

/// Prompt text → checked token ids for a serve request. What the client
/// caused (empty or oversized input) is an `invalid_request` on `input`;
/// what the tokenizer or model produced (a tokenizer fault, an id outside
/// the model vocabulary) is a server error.
pub(crate) fn encode_checked(
    tokenizer: &dyn Tokenize,
    prompt: &str,
    add_special_tokens: bool,
    vocab_size: u32,
    family: &str,
) -> Result<Vec<u32>, ServeError> {
    let ids = tokenizer
        .encode(prompt, add_special_tokens)
        .map_err(|error| match error {
            TokError::NativeInputTooLong { .. }
            | TokError::InputBytesTooLong(_)
            | TokError::InputTooLong(_) => {
                ServeError::invalid_request(Some("input"), format!("{family} prompt: {error}"))
            }
            error => ServeError::server_error(format!("tokenize {family} prompt: {error}")),
        })?;
    if ids.is_empty() {
        return Err(ServeError::invalid_request(
            Some("input"),
            format!("{family} prompt tokenized to zero tokens"),
        ));
    }
    ids.into_iter()
        .enumerate()
        .map(|(index, token)| {
            crate::checked_token_id(token, vocab_size, &format!("prompt[{index}]"))
                .map_err(|error| ServeError::server_error(error.to_string()))
        })
        .collect()
}

/// Forward budget for a session whose capacity is fixed at load.
pub(crate) fn required_forwards(
    family: &str,
    prompt_tokens: usize,
    max_tokens: usize,
    capacity: usize,
) -> Result<usize, ServeError> {
    if max_tokens == 0 {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            "max_output_tokens must be >= 1",
        ));
    }
    let required = prompt_tokens.checked_add(max_tokens - 1).ok_or_else(|| {
        ServeError::invalid_request(None, format!("{family} forward count overflow"))
    })?;
    if required > capacity {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            format!(
                "request needs {required} {family} forwards, beyond this server's capacity {capacity} (raise --max-context-tokens at startup)"
            ),
        ));
    }
    Ok(required)
}

/// Run the canonical loop. Every emitted token is written with `piece` and
/// followed by `tick`, because `piece` only touches the socket when the
/// output partition emits — a buffered tool block would otherwise hide a
/// disconnect for its whole length. A transport failure is `Aborted`; any
/// other failure is a server error. The stop token is counted, never
/// written. The backend finishes the outcome (family-specific capture and
/// stats happen between decode and finish).
pub(crate) fn decode_serial(
    request: DecodeRequest<'_>,
    sampler: &mut Sampler,
    tokenizer: &dyn Tokenize,
    sink: &mut dyn GenerationSink,
    mut forward: impl FnMut(u32) -> anyhow::Result<Vec<f32>>,
) -> Result<crate::GenerationResult, BackendFailure> {
    if request.max_tokens == 0 {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            "max_output_tokens must be >= 1",
        )
        .into());
    }
    let family = request.family;
    let vocab_size = request.vocab_size;
    let mut abort: Option<io::Error> = None;
    let generation = {
        let abort = &mut abort;
        crate::generate_serial(
            request.logits,
            request.max_tokens,
            request.stop_tokens,
            sampler,
            |token| {
                let bytes = tokenizer
                    .try_decode_piece_bytes(token)
                    .with_context(|| format!("decode {family} token {token}"))?;
                sink.piece(&bytes)
                    .and_then(|()| sink.tick())
                    .map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
            },
            |token| {
                let token = crate::checked_token_id(token, vocab_size, "generated")?;
                forward(token).with_context(|| format!("forward {family} token"))
            },
        )
    };
    generation.map_err(|error| match abort {
        Some(io_error) => BackendFailure::Aborted(io_error),
        None => ServeError::server_error(format!("{family} decode: {error:#}")).into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::sampling::SamplingConfig;

    /// Byte tokenizer: token = byte. `add_special` prepends 1 so the flag
    /// is observable.
    struct Bytes;
    impl Tokenize for Bytes {
        fn encode(&self, text: &str, add_special: bool) -> Result<Vec<i32>, TokError> {
            if text.len() > 8 {
                return Err(TokError::NativeInputTooLong {
                    bytes: text.len(),
                    max_bytes: 8,
                });
            }
            let mut ids: Vec<i32> = text.bytes().map(i32::from).collect();
            if add_special {
                ids.insert(0, 1);
            }
            Ok(ids)
        }
        fn try_decode_piece_bytes(&self, token: i32) -> Result<Vec<u8>, TokError> {
            if token == 255 {
                return Err(TokError::InvalidToken(token));
            }
            Ok(vec![u8::try_from(token).unwrap()])
        }
        fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
            Ok(String::from_utf8(self.try_decode_piece_bytes(token)?).unwrap())
        }
        fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
            tokens.iter().map(|&t| self.try_decode_piece(t)).collect()
        }
        fn decode(&self, tokens: &[i32]) -> String {
            self.try_decode(tokens).unwrap()
        }
        fn n_vocab(&self) -> u32 {
            256
        }
        fn bos(&self) -> Option<i32> {
            None
        }
        fn eos(&self) -> Option<i32> {
            Some(0)
        }
    }

    #[derive(Default)]
    struct Sink {
        pieces: Vec<u8>,
        ticks: usize,
        fail_piece_after: Option<usize>,
        fail_tick_after: Option<usize>,
    }
    impl GenerationSink for Sink {
        fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
            if self
                .fail_piece_after
                .is_some_and(|limit| self.pieces.len() >= limit)
            {
                return Err(io::Error::other("piece gone"));
            }
            self.pieces.extend_from_slice(bytes);
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            self.ticks += 1;
            if self.fail_tick_after.is_some_and(|limit| self.ticks > limit) {
                return Err(io::Error::other("tick gone"));
            }
            Ok(())
        }
    }

    fn greedy() -> Sampler {
        Sampler::new(SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        })
        .unwrap()
    }

    fn one_hot(token: usize) -> Vec<f32> {
        let mut logits = vec![0.0; 256];
        logits[token] = 10.0;
        logits
    }

    fn request<'a>(first: usize, max_tokens: usize, stop: &'a [i32]) -> DecodeRequest<'a> {
        DecodeRequest {
            family: "test",
            logits: one_hot(first),
            max_tokens,
            stop_tokens: stop,
            vocab_size: 256,
        }
    }

    /// Forward that yields a fixed script of next tokens.
    fn scripted(script: &[usize]) -> impl FnMut(u32) -> anyhow::Result<Vec<f32>> + '_ {
        let mut step = 0;
        move |_| {
            let logits = one_hot(script[step]);
            step += 1;
            Ok(logits)
        }
    }

    #[test]
    fn stops_on_a_stop_token_without_writing_it_and_ticks_per_piece() {
        let mut sink = Sink::default();
        let generation = decode_serial(
            request(b'a' as usize, 8, &[0, 9]),
            &mut greedy(),
            &Bytes,
            &mut sink,
            scripted(&[b'b' as usize, 0]),
        )
        .unwrap();
        assert_eq!(sink.pieces, b"ab");
        assert_eq!(sink.ticks, 2);
        assert_eq!(generation.tokens, vec![b'a' as i32, b'b' as i32, 0]);
        assert_eq!(generation.transitions, 2);
        assert_eq!(generation.stop_reason, crate::StopReason::Eos);
    }

    #[test]
    fn any_declared_stop_token_ends_generation_even_first() {
        let mut sink = Sink::default();
        let generation = decode_serial(
            request(9, 8, &[0, 9]),
            &mut greedy(),
            &Bytes,
            &mut sink,
            |_| unreachable!("no forward after an immediate stop"),
        )
        .unwrap();
        assert!(sink.pieces.is_empty());
        assert_eq!(sink.ticks, 0);
        assert_eq!(generation.tokens, vec![9]);
        assert_eq!(generation.transitions, 0);
    }

    #[test]
    fn max_tokens_bounds_generation_and_zero_is_a_client_error() {
        let mut sink = Sink::default();
        let generation = decode_serial(
            request(b'x' as usize, 3, &[0]),
            &mut greedy(),
            &Bytes,
            &mut sink,
            |_| Ok(one_hot(b'x' as usize)),
        )
        .unwrap();
        assert_eq!(sink.pieces, b"xxx");
        assert_eq!(generation.stop_reason, crate::StopReason::TokenLimit);

        let error = decode_serial(
            request(b'x' as usize, 0, &[0]),
            &mut greedy(),
            &Bytes,
            &mut Sink::default(),
            |_| unreachable!(),
        )
        .unwrap_err();
        let BackendFailure::Serve(error) = error else {
            panic!("{error:?}");
        };
        assert_eq!(error.status, 400);
        assert_eq!(error.param.as_deref(), Some("max_output_tokens"));
    }

    #[test]
    fn transport_failures_abort_and_model_failures_are_server_errors() {
        for sink in [
            Sink {
                fail_tick_after: Some(1),
                ..Sink::default()
            },
            Sink {
                fail_piece_after: Some(1),
                ..Sink::default()
            },
        ] {
            let mut sink = sink;
            let error = decode_serial(
                request(b'x' as usize, 8, &[0]),
                &mut greedy(),
                &Bytes,
                &mut sink,
                |_| Ok(one_hot(b'x' as usize)),
            )
            .unwrap_err();
            assert!(matches!(error, BackendFailure::Aborted(_)), "{error:?}");
        }

        let error = decode_serial(
            request(b'x' as usize, 8, &[0]),
            &mut greedy(),
            &Bytes,
            &mut Sink::default(),
            |_| anyhow::bail!("kernel fault"),
        )
        .unwrap_err();
        let BackendFailure::Serve(error) = error else {
            panic!("{error:?}");
        };
        assert_eq!(error.status, 500);
        assert!(error.message.contains("test decode"), "{}", error.message);
        assert!(error.message.contains("kernel fault"), "{}", error.message);

        // A tokenizer decode fault is a server error, not a transport abort.
        let error = decode_serial(
            request(255, 8, &[0]),
            &mut greedy(),
            &Bytes,
            &mut Sink::default(),
            |_| unreachable!(),
        )
        .unwrap_err();
        assert!(
            matches!(&error, BackendFailure::Serve(e) if e.status == 500),
            "{error:?}"
        );
    }

    #[test]
    fn encode_checked_classifies_client_and_server_faults() {
        assert_eq!(
            encode_checked(&Bytes, "ab", false, 256, "t").unwrap(),
            vec![97, 98]
        );
        assert_eq!(
            encode_checked(&Bytes, "ab", true, 256, "t").unwrap(),
            vec![1, 97, 98],
            "add_special_tokens must reach the tokenizer"
        );
        let empty = encode_checked(&Bytes, "", false, 256, "t").unwrap_err();
        assert_eq!((empty.status, empty.param.as_deref()), (400, Some("input")));
        let long = encode_checked(&Bytes, "123456789", false, 256, "t").unwrap_err();
        assert_eq!((long.status, long.param.as_deref()), (400, Some("input")));
        let oov = encode_checked(&Bytes, "ab", false, 97, "t").unwrap_err();
        assert_eq!(oov.status, 500);
    }

    #[test]
    fn required_forwards_budget() {
        assert_eq!(required_forwards("t", 10, 1, 10).unwrap(), 10);
        assert_eq!(required_forwards("t", 10, 3, 12).unwrap(), 12);
        assert!(required_forwards("t", 10, 0, 10).is_err());
        assert!(required_forwards("t", 10, 3, 11).is_err());
    }
}
