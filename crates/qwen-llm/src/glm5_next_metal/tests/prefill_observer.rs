//! Bounded actual-model oracle for the committed packed-chunk hook. The CLI
//! document test owns the 41,011-token checkpoint; here the same 51-token
//! starting position and 2048-row schedule cross the sparse frontier and end
//! in a three-token ragged chunk. No alternate execution/snapshot framework.
//!
//! Owner run (the test takes the internal production GPU lease):
//! MTL_DEBUG_LAYER=1 QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 \
//! GLM53_GGUF=/absolute/path/to/Maya.gguf cargo test -p qwen-llm --release --lib \
//! glm5_next_metal::packed::prefill_observer::committed_chunks_are_observational \
//! -- --exact --ignored --nocapture --test-threads=1

use super::*;
use crate::glm5_next_metal::tests::long_qualification_text;

fn assert_logit_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    for (channel, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "channel {channel}: {a} vs {b}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "channel {channel}: {a} vs {b}");
    }
}

#[test]
#[ignore = "actual-model state oracle: GLM53_GGUF, MTL_DEBUG_LAYER=1, internal production GPU lease"]
fn committed_chunks_are_observational() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    // Same internal lease as perf_lease(); that timing helper disallows Metal
    // validation, while this test intentionally requires it and records no timing.
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let path = std::path::PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF"));
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|token| token as u32)
        .collect();
    const PREFIX: usize = 51;
    const ROWS: usize = 2048;
    const CAPTURE: usize = PREFIX + ROWS;
    const END: usize = PREFIX + 2 * ROWS + 3;
    assert!(tokens.len() > END);
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, END + 1, ROWS).unwrap();
    session.set_packed_lineage(PackedLineage::Fast).unwrap();
    session.set_fast_precision(FastPrecision::Half).unwrap();
    session.prefill_packed(&ctx, &tokens[..PREFIX]).unwrap();
    let seed = session.capture_snapshot().unwrap();

    // Ordinary execution is the reference: unchanged 2048 + 2048 + 3 schedule,
    // head only on the last chunk, from exactly the same committed seed state.
    let expected_logits = session.prefill_packed(&ctx, &tokens[PREFIX..END]).unwrap();
    let expected_end = session.capture_snapshot().unwrap();
    let expected_decode = session.forward(&ctx, tokens[END]).unwrap();
    let expected_after_decode = session.capture_snapshot().unwrap();

    session.restore_snapshot(&seed).unwrap();
    let mut events = Vec::new();
    let mut checks = 0;
    let mut captured = None;
    let mut callback_end = None;
    let observed_logits = session
        .prefill_packed_with_observer(
            &ctx,
            &tokens[PREFIX..END],
            &mut || {
                checks += 1;
                Ok(())
            },
            &mut |committed, length| {
                assert!(!committed.is_poisoned());
                assert_eq!(
                    committed.position(),
                    PREFIX + events.iter().map(|&(_, n)| n).sum::<usize>() + length
                );
                events.push((committed.position(), length));
                if committed.position() == CAPTURE {
                    assert_eq!(length, ROWS);
                    captured = Some(committed.capture_snapshot().expect("committed full chunk"));
                }
                if committed.position() == END {
                    callback_end = Some(
                        committed
                            .capture_snapshot()
                            .expect("committed ragged chunk"),
                    );
                }
            },
        )
        .unwrap();
    assert_eq!(checks, 3);
    assert_eq!(
        events,
        [(CAPTURE, ROWS), (PREFIX + 2 * ROWS, ROWS), (END, 3)]
    );
    assert_logit_bits(&observed_logits, &expected_logits);
    // same_state checks every persistent byte: KDA conv/S, MLA visible latent,
    // pending ring and completed pools, plus position/lineage/precision.
    assert!(expected_end.same_state(&callback_end.unwrap()));
    assert!(expected_end.same_state(&session.capture_snapshot().unwrap()));
    assert_logit_bits(
        &session.forward(&ctx, tokens[END]).unwrap(),
        &expected_decode,
    );
    assert!(expected_after_decode.same_state(&session.capture_snapshot().unwrap()));

    // Restore the snapshot captured *inside* an existing full chunk boundary.
    // Continue the identical suffix with its original 2048 + 3 segmentation.
    let captured = captured.expect("full-chunk snapshot");
    assert_eq!(captured.position(), CAPTURE);
    session.restore_snapshot(&captured).unwrap();
    assert_logit_bits(
        &session.prefill_packed(&ctx, &tokens[CAPTURE..END]).unwrap(),
        &expected_logits,
    );
    assert!(expected_end.same_state(&session.capture_snapshot().unwrap()));
    assert_logit_bits(
        &session.forward(&ctx, tokens[END]).unwrap(),
        &expected_decode,
    );
    assert!(expected_after_decode.same_state(&session.capture_snapshot().unwrap()));

    // Cancellation before chunk one/two only observes the chunks already
    // committed, and leaves a capturable, unpoisoned boundary state.
    for stop_before in [1, 2] {
        session.restore_snapshot(&seed).unwrap();
        let mut checks = 0;
        let mut committed = Vec::new();
        let result = session.prefill_packed_with_observer(
            &ctx,
            &tokens[PREFIX..END],
            &mut || {
                checks += 1;
                if checks == stop_before {
                    Err("observer cancellation probe".into())
                } else {
                    Ok(())
                }
            },
            &mut |s, length| {
                assert!(!s.is_poisoned());
                committed.push((s.position(), length));
            },
        );
        assert!(matches!(result, Err(Glm5NextMetalError::Cancelled(_))));
        assert_eq!(checks, stop_before);
        assert_eq!(
            committed,
            if stop_before == 1 {
                vec![]
            } else {
                vec![(CAPTURE, ROWS)]
            }
        );
        assert!(!session.is_poisoned());
        assert!(
            (if stop_before == 1 { &seed } else { &captured })
                .same_state(&session.capture_snapshot().unwrap())
        );
    }

    // Whole-request validation rejects even a bad token in the final chunk
    // before cancellation checks or callbacks, with no partially advanced prefix.
    session.restore_snapshot(&seed).unwrap();
    let mut invalid_tokens = tokens[PREFIX..END].to_vec();
    *invalid_tokens.last_mut().unwrap() = weights.config.vocab_size;
    let result = session.prefill_packed_with_observer(
        &ctx,
        &invalid_tokens,
        &mut || panic!("invalid request reached cancellation check"),
        &mut |_, _| panic!("invalid request reached observer"),
    );
    assert!(matches!(result, Err(Glm5NextMetalError::Invalid(_))));
    assert!(seed.same_state(&session.capture_snapshot().unwrap()));

    // Existing fault injection makes a real sparse selector status fail after
    // GPU completion. No position is published, poison remains, no callback.
    let sparse_row = session.dense_rows_remaining();
    assert!(
        sparse_row < ROWS,
        "fixture must cross the sparse frontier in the first chunk"
    );
    session.corrupt_sparse_row = Some(sparse_row);
    let mut checks = 0;
    let result = session.prefill_packed_with_observer(
        &ctx,
        &tokens[PREFIX..END],
        &mut || {
            checks += 1;
            Ok(())
        },
        &mut |_, _| panic!("failed chunk reached observer"),
    );
    assert!(
        matches!(
            result,
            Err(Glm5NextMetalError::KernelValidation {
                stage: "packed sparse selection",
                ..
            })
        ),
        "{result:?}"
    );
    assert_eq!(checks, 1);
    assert_eq!(session.position(), PREFIX);
    assert!(session.is_poisoned());
    assert!(matches!(
        session.capture_snapshot(),
        Err(Glm5NextMetalError::Poisoned)
    ));
    drop(session);

    // No packed scratch: per-token cancellation and ordinary serial logits /
    // state are preserved, and even successful tokens never call the observer.
    let mut serial = Glm5NextSession::new(&ctx, &weights, 8).unwrap();
    let serial_seed = serial.capture_snapshot().unwrap();
    let expected = serial.prefill_packed(&ctx, &tokens[..3]).unwrap();
    let serial_end = serial.capture_snapshot().unwrap();
    serial.restore_snapshot(&serial_seed).unwrap();
    let mut checks = 0;
    let actual = serial
        .prefill_packed_with_observer(
            &ctx,
            &tokens[..3],
            &mut || {
                checks += 1;
                Ok(())
            },
            &mut |_, _| panic!("serial fallback reached observer"),
        )
        .unwrap();
    assert_eq!(checks, 3);
    assert_logit_bits(&actual, &expected);
    assert!(serial_end.same_state(&serial.capture_snapshot().unwrap()));
    serial.restore_snapshot(&serial_seed).unwrap();
    let mut checks = 0;
    let result = serial.prefill_packed_with_observer(
        &ctx,
        &tokens[..3],
        &mut || {
            checks += 1;
            if checks == 2 {
                Err("serial cancellation probe".into())
            } else {
                Ok(())
            }
        },
        &mut |_, _| panic!("serial cancellation reached observer"),
    );
    assert!(matches!(result, Err(Glm5NextMetalError::Cancelled(_))));
    assert_eq!(serial.position(), 1);
    assert!(!serial.is_poisoned());
}
