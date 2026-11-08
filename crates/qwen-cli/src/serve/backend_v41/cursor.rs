//! Compatibility path for V4.1; the lifecycle is shared with GLM.
pub(super) use crate::serve::decode_cursor::{Advance, DecodeCursor};

#[cfg(test)]
use super::super::decode_loop::DecodeRequest;
#[cfg(test)]
use super::super::http::{BackendFailure, GenerationSink};
#[cfg(test)]
use super::super::items::ServeError;
#[cfg(test)]
use qwen_llm::sampling::Sampler;
#[cfg(test)]
use qwen_llm::tokenizer::Tokenize;

#[cfg(test)]
mod tests {
    use super::super::super::decode_loop::decode_serial;
    use super::*;
    use qwen_llm::sampling::SamplingConfig;
    use qwen_llm::tokenizer::TokError;
    use std::io;

    /// Byte tokenizer: token = byte; 255 cannot be decoded.
    struct Bytes;
    impl Tokenize for Bytes {
        fn encode(&self, text: &str, _: bool) -> Result<Vec<i32>, TokError> {
            Ok(text.bytes().map(i32::from).collect())
        }
        fn try_decode_piece_bytes(&self, token: i32) -> Result<Vec<u8>, TokError> {
            if token == 255 {
                return Err(TokError::InvalidToken(token));
            }
            Ok(vec![u8::try_from(token).unwrap()])
        }
        fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
            Ok(String::from_utf8_lossy(&self.try_decode_piece_bytes(token)?).into_owned())
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

    #[derive(Debug, PartialEq, Clone)]
    enum Event {
        Piece(Vec<u8>),
        Tick,
        Forward(u32),
    }

    #[derive(Default)]
    struct Recorder {
        events: Vec<Event>,
        fail_piece_at: Option<usize>,
        fail_tick_at: Option<usize>,
        pieces: usize,
        ticks: usize,
    }
    impl GenerationSink for Recorder {
        fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.pieces += 1;
            if self.fail_piece_at == Some(self.pieces) {
                return Err(io::Error::other("piece gone"));
            }
            self.events.push(Event::Piece(bytes.to_vec()));
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            self.ticks += 1;
            if self.fail_tick_at == Some(self.ticks) {
                return Err(io::Error::other("tick gone"));
            }
            self.events.push(Event::Tick);
            Ok(())
        }
    }

    /// Deterministic pseudo-random logits so sampled decoding is exercised.
    fn logits_for(step: usize, peak: Option<usize>) -> Vec<f32> {
        let mut state = (step as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut logits: Vec<f32> = (0..256)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 1000) as f32 / 250.0
            })
            .collect();
        logits[0] = -10.0;
        logits[255] = -10.0;
        if let Some(peak) = peak {
            logits[peak] = 100.0;
        }
        logits
    }

    #[derive(Clone)]
    struct Case {
        sampling: SamplingConfig,
        first: Option<usize>,
        /// Forced next token after each forward (None = pseudo-random).
        script: Vec<Option<usize>>,
        max_tokens: usize,
        fail_forward_at: Option<usize>,
        fail_piece_at: Option<usize>,
        fail_tick_at: Option<usize>,
    }

    type Observed = (
        Vec<Event>,
        Result<(Vec<i32>, usize, crate::StopReason), (bool, u16, String)>,
    );

    fn classify(error: BackendFailure) -> (bool, u16, String) {
        match error {
            BackendFailure::Aborted(error) => (true, 0, error.to_string()),
            BackendFailure::Serve(error) => (false, error.status, error.message),
        }
    }

    fn forward_result(case: &Case, calls: usize) -> anyhow::Result<Vec<f32>> {
        if case.fail_forward_at == Some(calls) {
            anyhow::bail!("kernel fault at {calls}");
        }
        // Past the script, the model emits the stop token.
        Ok(logits_for(
            calls,
            case.script.get(calls - 1).copied().unwrap_or(Some(0)),
        ))
    }

    fn serial(case: &Case) -> Observed {
        let mut sink = Recorder {
            fail_piece_at: case.fail_piece_at,
            fail_tick_at: case.fail_tick_at,
            ..Recorder::default()
        };
        let mut forwarded = Vec::new();
        let mut calls = 0;
        let result = decode_serial(
            DecodeRequest {
                family: "V4.1",
                logits: logits_for(0, case.first),
                max_tokens: case.max_tokens,
                stop_tokens: &[0],
                vocab_size: 256,
            },
            &mut Sampler::new(case.sampling).unwrap(),
            &Bytes,
            &mut sink,
            |token| {
                calls += 1;
                let logits = forward_result(case, calls);
                if logits.is_ok() {
                    forwarded.push(token);
                }
                logits
            },
        );
        let mut events = sink.events;
        events.extend(forwarded.into_iter().map(Event::Forward));
        (
            events,
            result
                .map(|g| (g.tokens, g.transitions, g.stop_reason))
                .map_err(classify),
        )
    }

    fn cursor(case: &Case) -> Observed {
        let mut sink = Recorder {
            fail_piece_at: case.fail_piece_at,
            fail_tick_at: case.fail_tick_at,
            ..Recorder::default()
        };
        let mut sampler = Sampler::new(case.sampling).unwrap();
        let mut forwarded = Vec::new();
        let result = (|| {
            let mut cursor = DecodeCursor::new(DecodeRequest {
                family: "V4.1",
                logits: logits_for(0, case.first),
                max_tokens: case.max_tokens,
                stop_tokens: &[0],
                vocab_size: 256,
            })?;
            let mut calls = 0;
            loop {
                match cursor.advance(&mut sampler, None, &Bytes, &mut sink, || Ok(()))? {
                    Advance::Done(generation) => return Ok(generation),
                    Advance::Forward(token) => {
                        calls += 1;
                        let logits = forward_result(case, calls);
                        if logits.is_ok() {
                            forwarded.push(token);
                        }
                        cursor.on_logits(logits, 0.5, || Ok(()))?;
                    }
                }
            }
        })();
        let mut events = sink.events;
        events.extend(forwarded.into_iter().map(Event::Forward));
        (
            events,
            result
                .map(|g: crate::GenerationResult| (g.tokens, g.transitions, g.stop_reason))
                .map_err(classify),
        )
    }

    fn cases() -> Vec<Case> {
        let greedy = SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        };
        let sampled = SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.0,
            seed: 7,
        };
        let base = Case {
            sampling: greedy,
            first: Some(b'a' as usize),
            script: vec![Some(b'b' as usize), Some(b'c' as usize), Some(0)],
            max_tokens: 8,
            fail_forward_at: None,
            fail_piece_at: None,
            fail_tick_at: None,
        };
        vec![
            // EOS after three published tokens.
            base.clone(),
            // Immediate stop token: nothing published, no forward.
            Case {
                first: Some(0),
                ..base.clone()
            },
            // Token limit: the last token is published but never consumed.
            Case {
                max_tokens: 2,
                ..base.clone()
            },
            Case {
                max_tokens: 1,
                ..base.clone()
            },
            // Transport failures abort.
            Case {
                fail_piece_at: Some(2),
                ..base.clone()
            },
            Case {
                fail_tick_at: Some(3),
                ..base.clone()
            },
            // Model failure is a server error with decode_serial's message.
            Case {
                fail_forward_at: Some(2),
                ..base.clone()
            },
            // Tokenizer decode failure is a server error.
            Case {
                script: vec![Some(255)],
                ..base.clone()
            },
            // Seeded sampling over pseudo-random logits, to the token limit.
            Case {
                sampling: sampled,
                first: None,
                script: vec![None; 64],
                max_tokens: 40,
                ..base.clone()
            },
        ]
    }

    #[test]
    fn cursor_matches_decode_serial_events_tokens_and_errors() {
        for (index, case) in cases().iter().enumerate() {
            let expected = serial(case);
            let actual = cursor(case);
            assert_eq!(actual, expected, "case {index}");
        }
    }

    #[test]
    fn cursor_matches_decode_serial_counters() {
        let case = &cases()[0];
        let Ok((tokens, transitions, stop)) = cursor(case).1 else {
            panic!("generation");
        };
        assert_eq!(tokens, vec![b'a' as i32, b'b' as i32, b'c' as i32, 0]);
        assert_eq!(transitions, 3);
        assert_eq!(stop, crate::StopReason::Eos);
    }

    #[test]
    fn cursor_zero_max_tokens_is_a_client_error_and_checkpoint_failure_is_500() {
        let Err(BackendFailure::Serve(error)) = DecodeCursor::new(DecodeRequest {
            family: "V4.1",
            logits: vec![0.0; 256],
            max_tokens: 0,
            stop_tokens: &[0],
            vocab_size: 256,
        }) else {
            panic!("zero max_tokens must be refused");
        };
        assert_eq!(error.status, 400);
        let mut cursor = DecodeCursor::new(DecodeRequest {
            family: "V4.1",
            logits: logits_for(0, Some(1)),
            max_tokens: 4,
            stop_tokens: &[0],
            vocab_size: 256,
        })
        .unwrap();
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let Err(BackendFailure::Serve(error)) =
            cursor.advance(&mut sampler, None, &Bytes, &mut Recorder::default(), || {
                anyhow::bail!("termination signal")
            })
        else {
            panic!("checkpoint failure must fail the decode");
        };
        assert_eq!(error.status, 500);
        assert!(error.message.starts_with("V4.1 decode: termination signal"));
    }
}
