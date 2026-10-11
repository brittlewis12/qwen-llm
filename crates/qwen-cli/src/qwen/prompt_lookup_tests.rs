use super::*;

const PASSAGE: &str = "The storage service accepts a request, validates its key, and writes the record to a temporary file. Once the write is complete, it flushes the file and atomically renames it into place. Readers continue to see the previous complete record until the rename succeeds. A failed write leaves the old record unchanged. The caller receives a success response only after the new record is durable. To delete a record, the service first validates the key and then removes only the matching file. Unrelated records remain untouched. Every operation reports its own error rather than pretending that the operation succeeded.";

fn assert_position(sequence: &Sequence, expected: usize) {
    assert_eq!(sequence.position(), expected);
    assert!(
        sequence
            .metal_session()
            .kv_n_pos
            .iter()
            .all(|&p| p == expected)
    );
    let state = sequence.metal_session();
    let snapshot = state
        .snapshot(state.snapshot_identity(0, 0), vec![0; expected], None)
        .unwrap();
    for bytes in [&snapshot.gdn_state_arena, &snapshot.gdn_conv_arena] {
        assert!(
            bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        );
    }
    for bytes in [&snapshot.kv_k_arena, &snapshot.kv_v_arena] {
        let finite = match state.kv_k[0].dtype {
            GgmlType::F16 => bytes.chunks_exact(2).all(|b| {
                half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()
            }),
            GgmlType::F32 => bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            dtype => panic!("unexpected KV dtype: {dtype:?}"),
        };
        assert!(finite);
    }
}

fn finite_logits(logits: Vec<f32>) -> Vec<f32> {
    assert!(logits.iter().all(|x| x.is_finite()));
    logits
}

#[test]
#[ignore = "QWEN_PLD_TEST_MODEL=Qwen3.6-35B-A3B GGUF; Metal validation required"]
fn prompt_lookup_edit_then_next_turn() -> Result<()> {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let runtime = Runtime::new_metal()?;
    let loaded = runtime.load_model(std::env::var("QWEN_PLD_TEST_MODEL")?)?;
    assert_eq!(loaded.arch().kind, ArchKind::Moe);
    let model = qwen_llm::loader::Model::from_gguf(loaded.gguf())?;
    qwen_llm::metal_dflash::ensure_prompt_lookup_n8_supported_for_gguf(&model)
        .map_err(anyhow::Error::msg)?;
    qwen_llm::metal_dflash::ensure_prompt_lookup_n8_supported(loaded.metal_model())
        .map_err(anyhow::Error::msg)?;
    let tok = loaded.tokenizer()?;
    let prompt = messages::render_qwen_single_turn_prompt(
        &format!(
            "Copy the following text but replace every occurrence of 'record' with 'document'. Make no other changes and give no introduction or explanation: {PASSAGE}"
        ),
        None,
        messages::QwenGenerationMode::NoThinking,
    );
    let prompt_ids = tok.encode(&prompt, false)?;
    let stops = tok.encode("<|im_end|>", false)?;
    assert_eq!(stops.len(), 1);
    let mut reference = loaded.create_sequence(SequenceConfig::new(1024))?;
    let mut candidate = loaded.create_sequence(SequenceConfig::new(1024))?;
    let mut prefill =
        loaded.allocate_packed_prefill_scratch(loaded.plan_packed_prefill_scratch(16, 1024)?)?;
    let ref_logits = loaded.prefill(&mut reference, &mut prefill, &prompt_ids)?;
    let candidate_logits = loaded.prefill(&mut candidate, &mut prefill, &prompt_ids)?;
    let expected = generate_greedy(
        finite_logits(ref_logits),
        128,
        &stops,
        |_| Ok(()),
        |token| Ok(finite_logits(loaded.decode_token(&mut reference, token)?)),
    )?;
    let mut delivered = Vec::new();
    let result = generate_prompt_lookup(
        &loaded,
        &loaded.forward(),
        candidate,
        &prompt_ids,
        finite_logits(candidate_logits),
        128,
        &stops,
        |token| {
            delivered.push(token);
            Ok(())
        },
    )?;
    eprintln!(
        "PLD edit: verifies={} restores={} accepted={}/{}",
        result.stats.verify_calls,
        result.stats.restore_calls,
        result.stats.accepted_drafts,
        result.stats.drafts_scored
    );
    assert!(result.stats.accepted_drafts > 0);
    assert!(
        result.stats.restore_calls > 0,
        "fixture must exercise rejection/restore"
    );
    assert_eq!(result.generation.tokens, expected.tokens);
    assert_eq!(result.generation.stop_reason, StopReason::Eos);
    assert_eq!(delivered, expected.tokens[..expected.tokens.len() - 1]);
    let text: String = delivered.iter().map(|&id| tok.decode_piece(id)).collect();
    assert_eq!(text.trim(), PASSAGE.replace("record", "document"));
    candidate = result.sequence;
    let terminal_position = prompt_ids.len() + expected.tokens.len() - 1;
    assert_position(&reference, terminal_position);
    assert_position(&candidate, terminal_position);

    // Generation leaves its final emitted token unconsumed. Append EOS exactly
    // once, then a rendered user turn to the actual retained sequences.
    let terminal = *expected.tokens.last().unwrap();
    finite_logits(loaded.decode_token(&mut reference, terminal)?);
    finite_logits(loaded.decode_token(&mut candidate, terminal)?);
    let suffix = format!(
        "\n{}",
        messages::render_qwen_single_turn_prompt(
            "Which operation leaves unrelated documents untouched? Answer with the sentence describing how to delete a document from your previous response, and nothing else.",
            None,
            messages::QwenGenerationMode::NoThinking,
        )
    );
    let suffix_ids = tok.encode(&suffix, false)?;
    let ref_logits = finite_logits(loaded.prefill(&mut reference, &mut prefill, &suffix_ids)?);
    let candidate_logits =
        finite_logits(loaded.prefill(&mut candidate, &mut prefill, &suffix_ids)?);
    let delta =
        qwen_llm::compare::report_max_abs_diff_f32(ref_logits.iter(), candidate_logits.iter());
    eprintln!("next-turn prompt logit max_abs={delta:e}");
    let next_position = terminal_position + 1 + suffix_ids.len();
    assert_position(&reference, next_position);
    assert_position(&candidate, next_position);
    let next_ref = generate_greedy(
        ref_logits,
        96,
        &stops,
        |_| Ok(()),
        |token| Ok(finite_logits(loaded.decode_token(&mut reference, token)?)),
    )?;
    let next_candidate = generate_greedy(
        candidate_logits,
        96,
        &stops,
        |_| Ok(()),
        |token| Ok(finite_logits(loaded.decode_token(&mut candidate, token)?)),
    )?;
    let text: String = next_candidate
        .tokens
        .iter()
        .filter(|id| !stops.contains(id))
        .map(|&id| tok.decode_piece(id))
        .collect();
    eprintln!("next-turn response: {text}");
    assert_eq!(next_candidate.tokens, next_ref.tokens);
    assert_eq!(next_candidate.stop_reason, StopReason::Eos);
    assert!(text.contains("validates the key"));
    assert!(text.contains("matching file"));
    assert_position(&reference, next_position + next_ref.tokens.len() - 1);
    assert_position(&candidate, reference.position());
    Ok(())
}
