//! One synchronous prefill/decode loop; tests inject only the step function.
use anyhow::{Result, ensure};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub(super) struct Outcome {
    pub tokens: Vec<u32>,
    pub stop: Option<u32>,
}

pub(super) fn finish_text(
    tokenizer: &tokenizers::Tokenizer,
    ids: &[u32],
    emitted: &str,
) -> Result<String> {
    let complete = tokenizer
        .decode(ids, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    ensure!(
        complete.starts_with(emitted),
        "incremental tokenizer output is not a prefix of complete output"
    );
    Ok(complete[emitted.len()..].to_owned())
}

pub(super) fn required_context(
    prompt: &[u32],
    maximum: usize,
    context: usize,
    vocab: usize,
) -> Result<usize> {
    ensure!(!prompt.is_empty(), "prompt must contain at least one token");
    ensure!(maximum > 0, "max tokens must be positive");
    ensure!(
        prompt.iter().all(|&id| (id as usize) < vocab),
        "prompt token outside vocabulary"
    );
    let required = prompt
        .len()
        .checked_add(maximum - 1)
        .ok_or_else(|| anyhow::anyhow!("context extent overflow"))?;
    ensure!(
        required <= context,
        "prompt plus decode requires {required} forwards, context is {context}"
    );
    Ok(required)
}

pub(super) fn run(
    prompt: &[u32],
    maximum: usize,
    vocab: usize,
    stops: &[u32],
    sampling: SamplingConfig,
    cancel: &AtomicBool,
    mut step: impl FnMut(u32) -> Result<Vec<f32>>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Outcome> {
    let checkpoint = || {
        ensure!(!cancel.load(Ordering::Acquire), "V4.1 generation cancelled");
        Ok(())
    };
    ensure!(
        !prompt.is_empty() && maximum > 0,
        "empty prompt or zero output limit"
    );
    let mut sampler = Sampler::new(sampling)?;
    let mut logits = Vec::new();
    for &token in prompt {
        checkpoint()?;
        logits = step(token)?;
    }
    let mut tokens = Vec::new();
    for index in 0..maximum {
        checkpoint()?;
        ensure!(
            logits.len() == vocab,
            "forward returned wrong vocabulary size"
        );
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "forward returned non-finite logits"
        );
        let token = u32::try_from(sampler.sample(&logits)?.token)?;
        tokens.push(token);
        if stops.contains(&token) {
            return Ok(Outcome {
                tokens,
                stop: Some(token),
            });
        }
        emit(token)?;
        if index + 1 < maximum {
            checkpoint()?;
            logits = step(token)?;
        }
    }
    Ok(Outcome { tokens, stop: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v41_tokenizer_stream_reassembles_utf8_and_flushes_an_unfinished_tail() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/v41_engram_token_map.json"
        ))
        .unwrap();
        let tokenizer =
            tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&fixture["tokenizer"]).unwrap())
                .unwrap();
        let first = tokenizer.token_to_id("\u{c3}").unwrap();
        let second = tokenizer.token_to_id("\u{a9}").unwrap();
        let mut decoder = tokenizer.decode_stream(false);
        assert_eq!(decoder.step(first).unwrap(), None);
        assert_eq!(decoder.step(second).unwrap().as_deref(), Some("\u{e9}"));
        assert_eq!(
            finish_text(&tokenizer, &[first, second], "\u{e9}").unwrap(),
            ""
        );
        assert_eq!(finish_text(&tokenizer, &[first], "").unwrap(), "\u{fffd}");
        assert!(finish_text(&tokenizer, &[first, second], "wrong").is_err());
    }
    #[test]
    fn v41_decode_prefills_once_emits_first_token_and_stops_without_extra_forward() {
        let cancel = AtomicBool::new(false);
        let mut consumed = Vec::new();
        let mut output = Vec::new();
        let result = run(
            &[0, 2],
            4,
            3,
            &[1],
            SamplingConfig::default(),
            &cancel,
            |id| {
                consumed.push(id);
                Ok(if consumed.len() < 3 {
                    vec![0., 0., 5.]
                } else {
                    vec![0., 5., 0.]
                })
            },
            |id| {
                output.push(id);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(consumed, [0, 2, 2]);
        assert_eq!(output, [2]);
        assert_eq!(result.tokens, [2, 1]);
        assert_eq!(result.stop, Some(1));
        let mut calls = 0;
        run(
            &[0],
            1,
            3,
            &[1],
            SamplingConfig::default(),
            &cancel,
            |_| {
                calls += 1;
                Ok(vec![0., 0., 1.])
            },
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn v41_decode_rejects_overflow_bad_logits_and_honors_cancellation() {
        assert!(required_context(&[], 1, 8, 3).is_err());
        assert!(required_context(&[3], 1, 8, 3).is_err());
        assert!(required_context(&[0, 1], usize::MAX, usize::MAX, 3).is_err());
        assert!(required_context(&[0, 1], 3, 3, 3).is_err());
        assert!(required_context(&[0, 1], 2, 3, 3).is_ok());
        assert_eq!(required_context(&[0, 1], 2, 1_048_576, 3).unwrap(), 3);
        let cancel = AtomicBool::new(false);
        assert!(
            run(
                &[0],
                1,
                3,
                &[1],
                SamplingConfig::default(),
                &cancel,
                |_| Ok(vec![f32::NAN; 3]),
                |_| Ok(())
            )
            .is_err()
        );
        let mut calls = 0;
        assert!(
            run(
                &[0],
                2,
                3,
                &[1],
                SamplingConfig::default(),
                &cancel,
                |_| {
                    calls += 1;
                    Ok(vec![0., 0., 1.])
                },
                |_| {
                    cancel.store(true, Ordering::Release);
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(calls, 1);
    }
}
