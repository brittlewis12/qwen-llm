use super::*;

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, GgmlType::F32);
    unsafe {
        let pointer = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
    }
}

fn assert_bits(label: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{label}: length");
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            left.is_finite() && right.is_finite(),
            "{label}[{index}]: nonfinite"
        );
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{label}[{index}]: {left} != {right}"
        );
    }
}

#[test]
#[ignore = "QWEN_CAPTURE_TEST_MODEL; real dense or MoE model with MTL_DEBUG_LAYER=1"]
fn scalar_capture_preserves_serial_state_and_continuation() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = std::env::var("QWEN_CAPTURE_TEST_MODEL").unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mm = MetalModel::load(&ctx, &gguf, &model).unwrap();
    let forward = MetalForward::new(&ctx, &mm);
    let serial = |token, position, session: &mut MetalSession| {
        if mm.arch.kind == ArchKind::Moe {
            forward
                .single_token_profiled_moe(token, position, session)
                .unwrap()
                .0
        } else {
            forward.single_token(token, position, session).unwrap()
        }
    };
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens = tokenizer
        .encode("The quick brown fox jumps over the lazy dog", false)
        .unwrap();
    assert!(tokens.len() > 2 && mm.blocks.len() > 4);
    let h = mm.arch.hidden_size as usize;
    let last = mm.blocks.len() as u32 - 1;
    let layers = [3, last, 0, 3];
    let reordered = [last, 3, 0, 3];
    let elements = layers.len() * h;
    let full_dst = MetalTensor::zeros_f32(&ctx, vec![elements as u64]).unwrap();
    // Exercise a nonzero offset, as DFlash passes capture-slot views.
    let skip_storage = MetalTensor::zeros_f32(&ctx, vec![(elements + 2 * h) as u64]).unwrap();
    let skip_dst = skip_storage.view_subrange(h as u64, vec![elements as u64]);
    let mut reference = MetalSession::fresh(&ctx, &mm, tokens.len() + 4).unwrap();
    let mut default = MetalSession::fresh(&ctx, &mm, tokens.len() + 4).unwrap();
    let mut default_max_delta = 0.0f32;
    let mut full = MetalSession::fresh(&ctx, &mm, tokens.len() + 4).unwrap();
    let mut skip = MetalSession::fresh(&ctx, &mm, tokens.len() + 4).unwrap();
    for (position, &token) in tokens.iter().enumerate() {
        let expected = serial(token, position as u32, &mut reference);
        let default_logits = forward
            .single_token(token, position as u32, &mut default)
            .unwrap();
        default_max_delta = default_max_delta.max(max_abs_finite(
            "serial/default logits",
            &expected,
            &default_logits,
        ));
        assert_eq!(
            argmax_i32_local(&expected),
            argmax_i32_local(&default_logits)
        );
        let dot: f64 = expected
            .iter()
            .zip(&default_logits)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        assert!(dot / (norm(&expected) * norm(&default_logits)) > 0.99999);
        let captured = forward
            .single_token_with_multi_hidden(token, position as u32, &mut full, &layers, &full_dst)
            .unwrap();
        assert_bits("capture logits", &captured, &expected);
        if position + 1 < tokens.len() {
            let prior_logits = read_f32(&skip.logits);
            forward
                .single_token_with_multi_hidden_no_tail(
                    token,
                    position as u32,
                    &mut skip,
                    &reordered,
                    &skip_dst,
                )
                .unwrap();
            assert_bits(
                "no-tail leaves logits untouched",
                &read_f32(&skip.logits),
                &prior_logits,
            );
        } else {
            let logits = forward
                .single_token_with_multi_hidden(
                    token,
                    position as u32,
                    &mut skip,
                    &reordered,
                    &skip_dst,
                )
                .unwrap();
            assert_bits("no-tail final logits", &logits, &expected);
        }
        let captured = read_f32(&full_dst);
        let skipped = read_f32(&skip_dst);
        assert_bits(
            "last block residual",
            &captured[h..2 * h],
            &read_f32(&reference.x),
        );
        assert_bits("duplicate layers", &captured[..h], &captured[3 * h..]);
        for (slot, other) in [1, 0, 2, 3].into_iter().enumerate() {
            assert_bits(
                "caller layer order",
                &captured[slot * h..(slot + 1) * h],
                &skipped[other * h..(other + 1) * h],
            );
        }
        assert_eq!(full.kv_n_pos, reference.kv_n_pos);
        assert_eq!(skip.kv_n_pos, reference.kv_n_pos);
    }
    let guard = read_f32(&skip_storage);
    assert!(
        guard[..h]
            .iter()
            .chain(&guard[elements + h..])
            .all(|&x| x == 0.0)
    );
    let identity = reference.snapshot_identity(1, 2);
    let expected = reference
        .snapshot(identity.clone(), tokens.clone(), None)
        .unwrap();
    for session in [&full, &skip] {
        let captured = session
            .snapshot(identity.clone(), tokens.clone(), None)
            .unwrap();
        assert_eq!(captured.kv_n_pos, expected.kv_n_pos);
        assert!(captured.kv_k_arena == expected.kv_k_arena, "K payload");
        assert!(captured.kv_v_arena == expected.kv_v_arena, "V payload");
        assert!(
            captured.gdn_conv_arena == expected.gdn_conv_arena,
            "conv payload"
        );
        assert!(
            captured.gdn_state_arena == expected.gdn_state_arena,
            "GDN payload"
        );
    }
    for offset in 0..3 {
        let position = (tokens.len() + offset) as u32;
        let token = tokens[offset];
        let expected = forward
            .single_token(token, position, &mut reference)
            .unwrap();
        assert_bits(
            "full-capture continuation",
            &forward.single_token(token, position, &mut full).unwrap(),
            &expected,
        );
        assert_bits(
            "no-tail continuation",
            &forward.single_token(token, position, &mut skip).unwrap(),
            &expected,
        );
    }

    // Off-mode can begin from a prefix produced by the default dispatch, not
    // just from a fresh serial history. Use the real DFlash capture-slot shape.
    let handoff = default
        .snapshot(identity.clone(), tokens.clone(), None)
        .unwrap();
    for session in [&mut reference, &mut full, &mut skip] {
        *session = MetalSession::fresh(&ctx, &mm, tokens.len() + 4).unwrap();
        session.restore_from(&handoff, &identity).unwrap();
    }
    let slots =
        crate::metal_dflash::MetalDFlashVerifyScratch::fresh(&ctx, &mm, 2, layers.len() as u32)
            .unwrap();
    let slot = slots.hidden_capture_n_slot(1);
    let position = tokens.len() as u32;
    let expected = serial(tokens[0], position, &mut reference);
    assert_bits(
        "restored capture logits",
        &forward
            .single_token_with_multi_hidden(tokens[0], position, &mut full, &layers, &full_dst)
            .unwrap(),
        &expected,
    );
    forward
        .single_token_with_multi_hidden_no_tail(tokens[0], position, &mut skip, &layers, &slot)
        .unwrap();
    assert_bits(
        "restored capture slots",
        &read_f32(&full_dst),
        &read_f32(&slot),
    );
    let mut prefix = tokens.clone();
    prefix.push(tokens[0]);
    let expected = reference
        .snapshot(identity.clone(), prefix.clone(), None)
        .unwrap();
    for session in [&full, &skip] {
        let actual = session
            .snapshot(identity.clone(), prefix.clone(), None)
            .unwrap();
        assert_eq!(actual.kv_n_pos, expected.kv_n_pos);
        assert!(actual.kv_k_arena == expected.kv_k_arena);
        assert!(actual.kv_v_arena == expected.kv_v_arena);
        assert!(actual.gdn_conv_arena == expected.gdn_conv_arena);
        assert!(actual.gdn_state_arena == expected.gdn_state_arena);
    }
    let expected = forward
        .single_token(tokens[1], position + 1, &mut reference)
        .unwrap();
    for session in [&mut full, &mut skip] {
        assert_bits(
            "restored default continuation",
            &forward
                .single_token(tokens[1], position + 1, session)
                .unwrap(),
            &expected,
        );
    }

    let empty = MetalTensor::zeros_f32(&ctx, vec![0]).unwrap();
    let mut empty_session = MetalSession::fresh(&ctx, &mm, 2).unwrap();
    let mut ordinary = MetalSession::fresh(&ctx, &mm, 2).unwrap();
    forward
        .single_token_with_multi_hidden_no_tail(tokens[0], 0, &mut empty_session, &[], &empty)
        .unwrap();
    serial(tokens[0], 0, &mut ordinary);
    assert_bits(
        "empty capture continuation",
        &forward
            .single_token_with_multi_hidden(tokens[1], 1, &mut empty_session, &[], &empty)
            .unwrap(),
        &serial(tokens[1], 1, &mut ordinary),
    );

    let mut invalid = MetalSession::fresh(&ctx, &mm, 2).unwrap();
    let alias = invalid.x.clone();
    let wrong_dtype = MetalTensor::zeros_f16(&ctx, vec![h as u64]).unwrap();
    let valid = MetalTensor::zeros_f32(&ctx, vec![h as u64]).unwrap();
    for run_tail in [true, false] {
        for (layers, destination) in [
            (&[0][..], &alias),
            (&[0][..], &wrong_dtype),
            (&[last + 1][..], &valid),
        ] {
            let result = if run_tail {
                forward
                    .single_token_with_multi_hidden(tokens[0], 0, &mut invalid, layers, destination)
                    .map(|_| ())
            } else {
                forward.single_token_with_multi_hidden_no_tail(
                    tokens[0],
                    0,
                    &mut invalid,
                    layers,
                    destination,
                )
            };
            assert!(matches!(
                result,
                Err(MfError::Metal(MetalError::BadShape { .. }))
            ));
            assert!(invalid.kv_n_pos.iter().all(|&position| position == 0));
        }
    }
    eprintln!(
        "scalar capture {:?}: full/no-tail, caller order, serial state, continuation, and validation passed; default decode max logit delta={default_max_delta:e}",
        mm.arch.kind
    );
}
