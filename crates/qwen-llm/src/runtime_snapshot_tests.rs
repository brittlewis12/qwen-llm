//! Real runtime boundary witnesses, not snapshot-reuse or performance promotion.

use super::*;
use crate::metal::{BlitEncoder, MetalTensor};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

fn checkpoint(model: &LoadedModel, sequence: &Sequence, tokens: &[i32]) -> PreparedCheckpoint {
    model
        .prepare_checkpoint_boundary(sequence, tokens.to_vec(), None, None, None, 0)
        .unwrap()
}

fn replace_first_word(ctx: &MetalContext, destination: &MetalTensor, bytes: [u8; 4]) {
    let source =
        MetalTensor::from_bytes(ctx, &bytes, vec![1], crate::tensor::GgmlType::F32).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = BlitEncoder::begin(&command);
    encoder.copy_buffer(
        &source.buffer,
        source.offset,
        &destination.buffer,
        destination.offset,
        4,
    );
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
}

#[test]
#[ignore = "serial Metal, real 0.8B runtime snapshot alias witness"]
fn runtime_snapshot_observes_alias_retained_across_restore() {
    let runtime = Runtime::metal().unwrap();
    let model = runtime
        .load_model(crate::test_fixtures::QWEN35_0_8B_F32.path())
        .unwrap();
    let mut producer = model.create_sequence(SequenceConfig::new(8)).unwrap();
    let tokens = [1, 2];
    for token in tokens {
        model.decode_token(&mut producer, token).unwrap();
    }
    let root = checkpoint(&model, &producer, &tokens);
    let original = root.snapshot.kv_k_arena.clone();
    let bytes: [u8; 4] = original[..4].try_into().unwrap();
    let replacement = bytes.map(|b| b ^ 0x5a);
    for escape_before_restore in [true, false] {
        let mut sequence = model.create_sequence(SequenceConfig::new(8)).unwrap();
        let early_alias = escape_before_restore.then(|| sequence.metal_session().kv_k[0].clone());
        model
            .restore_prepared_checkpoint(&root, &mut sequence, &tokens)
            .unwrap();
        let alias = early_alias.unwrap_or_else(|| sequence.metal_session().kv_k[0].clone());
        replace_first_word(model.context(), &alias, replacement);
        let captured = checkpoint(&model, &sequence, &tokens);
        let mut expected = original.clone();
        expected[..4].copy_from_slice(&replacement);
        assert_eq!(captured.snapshot.kv_k_arena, expected);
        assert_eq!(captured.snapshot.kv_v_arena, root.snapshot.kv_v_arena);
        assert_eq!(
            captured.snapshot.gdn_state_arena,
            root.snapshot.gdn_state_arena
        );
        assert_eq!(
            captured.snapshot.gdn_conv_arena,
            root.snapshot.gdn_conv_arena
        );
        assert_eq!(root.snapshot.kv_k_arena, original);
        assert_eq!(sequence.position(), tokens.len());
        assert_ne!(
            captured.snapshot.kv_k_arena, root.snapshot.kv_k_arena,
            "naive restored-prefix reuse would silently discard the escaped-alias write"
        );
    }
}

fn assert_same_payload(a: &PreparedCheckpoint, b: &PreparedCheckpoint) {
    assert_eq!(a.snapshot.kv_k_arena, b.snapshot.kv_k_arena);
    assert_eq!(a.snapshot.kv_v_arena, b.snapshot.kv_v_arena);
    assert_eq!(a.snapshot.gdn_state_arena, b.snapshot.gdn_state_arena);
    assert_eq!(a.snapshot.gdn_conv_arena, b.snapshot.gdn_conv_arena);
    assert_eq!(a.snapshot.kv_n_pos, b.snapshot.kv_n_pos);
}

#[test]
#[ignore = "serial Metal, real 0.8B packed capture destination guards"]
fn runtime_packed_capture_rejects_session_and_scratch_aliases() {
    let runtime = Runtime::metal().unwrap();
    let model = runtime
        .load_model(crate::test_fixtures::QWEN35_0_8B_F32.path())
        .unwrap();
    let mut sequence = model.create_sequence(SequenceConfig::new(8)).unwrap();
    model.decode_token(&mut sequence, 1).unwrap();
    let root = checkpoint(&model, &sequence, &[1]);
    let h = u64::from(model.arch().hidden_size);
    let plan = model.plan_packed_prefill_scratch(2, 8).unwrap();
    let mut scratch = model.allocate_packed_prefill_scratch(plan).unwrap();
    let aliases = [
        sequence.metal_session().x.clone(),
        sequence.metal_session().gdn_state[0].view_subrange(1, vec![h]),
        sequence.metal_session().kv_k[0].view_subrange(0, vec![h]),
        scratch.x_pack.view_subrange(h, vec![h]),
    ];
    for destination in &aliases {
        let error = prefill_tokens_with_multi_hidden(
            &model.forward(),
            &[2],
            1,
            &mut sequence.state,
            &mut scratch.inner,
            &[0],
            Some(destination),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("aliases mutable session or prefill scratch")
        );
        assert_eq!(sequence.position(), 1);
        assert_same_payload(&root, &checkpoint(&model, &sequence, &[1]));
    }

    let mut plain = model.create_sequence(SequenceConfig::new(8)).unwrap();
    model
        .restore_prepared_checkpoint(&root, &mut plain, &[1, 2])
        .unwrap();
    let plain_logits = model.prefill(&mut plain, &mut scratch, &[2]).unwrap();
    let destination = MetalTensor::zeros_f32(model.context(), vec![h]).unwrap();
    let captured_logits = model
        .prefill_with_hidden_capture(&mut sequence, &mut scratch, &[2], &[0], &destination)
        .unwrap();
    assert_eq!(
        plain_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        captured_logits
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_same_payload(
        &checkpoint(&model, &plain, &[1, 2]),
        &checkpoint(&model, &sequence, &[1, 2]),
    );

    let escaped = sequence.metal_session().x.clone();
    let error = model
        .prefill_with_hidden_capture(&mut sequence, &mut scratch, &[3], &[0], &escaped)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("aliases mutable session or prefill scratch")
    );
    assert_eq!(sequence.position(), 2);
    assert!(
        sequence.state.ensure_usable().is_err(),
        "owned failed prefill must remain fail-stop"
    );
}

#[test]
#[ignore = "serial Metal, real 0.8B owned serial prompt append equivalence"]
fn runtime_owned_prompt_token_matches_raw_append_and_fails_closed() {
    let runtime = Runtime::metal().unwrap();
    let model = runtime
        .load_model(crate::test_fixtures::QWEN35_0_8B_F32.path())
        .unwrap();
    let mut raw = model.create_sequence(SequenceConfig::new(4)).unwrap();
    for token in [1, 2] {
        model.decode_token(&mut raw, token).unwrap();
    }
    let root = checkpoint(&model, &raw, &[1, 2]);
    let mut owned = model.create_sequence(SequenceConfig::new(4)).unwrap();
    model
        .restore_prepared_checkpoint(&root, &mut owned, &[1, 2, 3, 4])
        .unwrap();
    model
        .forward()
        .single_token_no_tail(3, 2, unsafe { raw.metal_session_mut() })
        .unwrap();
    raw.advance_by(1).unwrap();
    model.prefill_token_prompt_only(&mut owned, 3).unwrap();
    assert_eq!(owned.position(), 3);
    assert_same_payload(
        &checkpoint(&model, &raw, &[1, 2, 3]),
        &checkpoint(&model, &owned, &[1, 2, 3]),
    );
    let raw_logits = model
        .forward()
        .single_token(4, 3, unsafe { raw.metal_session_mut() })
        .unwrap();
    raw.advance_by(1).unwrap();
    let owned_logits = model.decode_token(&mut owned, 4).unwrap();
    assert_eq!(
        raw_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        owned_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(owned.position(), 4);
    let full = checkpoint(&model, &owned, &[1, 2, 3, 4]);
    assert_same_payload(&checkpoint(&model, &raw, &[1, 2, 3, 4]), &full);
    assert!(matches!(
        model.prefill_token_prompt_only(&mut owned, 5),
        Err(RuntimeError::SequenceCapacityExceeded { .. })
    ));
    assert_same_payload(&full, &checkpoint(&model, &owned, &[1, 2, 3, 4]));

    let mut foreign = model.create_sequence(SequenceConfig::new(4)).unwrap();
    foreign.owner = Arc::new(ModelOwnerToken::new());
    assert!(matches!(
        model.prefill_token_prompt_only(&mut foreign, 1),
        Err(RuntimeError::SequenceModelMismatch)
    ));
    assert_eq!(foreign.position(), 0);
    assert!(foreign.state.ensure_usable().is_ok());

    let mut invalid = model.create_sequence(SequenceConfig::new(4)).unwrap();
    assert!(model.prefill_token_prompt_only(&mut invalid, -1).is_err());
    assert_eq!(invalid.position(), 0);
    assert!(invalid.state.ensure_usable().is_err());
    assert!(model.decode_token(&mut invalid, 1).is_err());
    assert!(
        model
            .prepare_checkpoint_boundary(&invalid, vec![], None, None, None, 0)
            .is_err()
    );
}
