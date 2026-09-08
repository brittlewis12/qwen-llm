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
