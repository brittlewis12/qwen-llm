use super::*;

fn identity_bytes() -> Vec<u8> {
    let mut bytes = vec![0; MATRIX_BYTES];
    for i in 0..WIDTH {
        set(&mut bytes, i, i, 1.0);
    }
    bytes
}

fn set(bytes: &mut [u8], target: usize, source: usize, value: f32) {
    let offset = (target * WIDTH + source) * 2;
    bytes[offset..offset + 2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
}

#[test]
fn typed_matrix_is_exact_finite_bounded_and_byte_preserving() {
    assert_eq!(MATRIX_BYTES, 32 * 1024 * 1024);
    for length in [0, 2, MATRIX_BYTES - 1, MATRIX_BYTES + 1] {
        assert!(K2LinearF16::from_target_source_le(vec![0; length]).is_err());
    }
    for bits in [0x7c00u16, 0xfc00, 0x7e01, 0xfe01] {
        let mut bytes = vec![0; MATRIX_BYTES];
        bytes[MATRIX_BYTES - 2..].copy_from_slice(&bits.to_le_bytes());
        assert!(K2LinearF16::from_target_source_le(bytes).is_err());
    }
    let mut bytes = identity_bytes();
    set(&mut bytes, 0, 1, -2.0);
    bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
    bytes[6..8].copy_from_slice(&0x8000u16.to_le_bytes());
    let expected = bytes.clone();
    let matrix = K2LinearF16::from_target_source_le(bytes).unwrap();
    assert_eq!(matrix.byte_len(), MATRIX_BYTES);
    assert_eq!(matrix.bytes, expected);
    assert_eq!(
        u16::from_le_bytes(matrix.bytes[2..4].try_into().unwrap()),
        half::f16::from_f32(-2.0).to_bits()
    );
    assert_eq!(
        u16::from_le_bytes(matrix.bytes[WIDTH * 2..WIDTH * 2 + 2].try_into().unwrap()),
        0
    );
}

fn cache_bits(session: &K2Session<'_, '_>) -> Vec<u16> {
    // Private shared F16 storage, checked at construction; synchronous commands.
    unsafe {
        std::slice::from_raw_parts(
            session
                .buffers
                .cache
                .buffer
                .contents()
                .as_ptr()
                .cast::<u16>(),
            session.buffers.cache.n_elements() as usize,
        )
        .to_vec()
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "full checkpoint GPU linear readout correctness; production lease and K2_GGUF required"]
fn gpu_linear_readout_orientation_identity_isolation_and_poison() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 4).unwrap();
    let mut plain = model.create_session(37).unwrap();
    let capture = plain.append_with_captures(&[0, 42, 17], &[35]).unwrap();
    let original_cache = cache_bits(&plain);
    let continuation = plain.append(&[19]).unwrap();
    drop(plain);

    let matrix = K2LinearF16::from_target_source_le(identity_bytes()).unwrap();
    let mut session = model.create_session(37).unwrap();
    session.append(&[0, 42, 17]).unwrap();
    let identity = session
        .readout_linear_f16(&matrix, &capture.residuals)
        .unwrap();
    assert_eq!(bits(&identity.residual), bits(&capture.residuals));
    assert_eq!(bits(&identity.logits), bits(&capture.logits));
    assert_eq!(cache_bits(&session), original_cache);
    assert_eq!(session.committed_len(), 3);
    drop(matrix);

    let mut bytes = identity_bytes();
    set(&mut bytes, 0, 1, 2.0);
    set(&mut bytes, 1, 0, -3.0);
    set(&mut bytes, 2, 3, 4.0);
    let matrix = K2LinearF16::from_target_source_le(bytes).unwrap();
    let mut input = vec![0.125; WIDTH];
    input[..4].copy_from_slice(&[0.25, -0.5, 0.75, -1.25]);
    let mut expected = input.clone();
    expected[0] = input[0] + 2.0 * input[1];
    expected[1] = input[1] - 3.0 * input[0];
    expected[2] = input[2] + 4.0 * input[3];
    let output = session.readout_linear_f16(&matrix, &input).unwrap();
    assert_eq!(bits(&output.residual), bits(&expected));
    assert_ne!(
        output.residual[0],
        input[0] - 3.0 * input[1],
        "transpose negative control"
    );
    assert_eq!(
        bits(&output.logits),
        bits(&session.readout(&expected).unwrap())
    );
    assert_eq!(session.committed_len(), 3);
    assert_eq!(cache_bits(&session), original_cache);
    for row in [
        vec![],
        vec![0.0; WIDTH - 1],
        vec![f32::NAN; WIDTH],
        vec![f32::INFINITY; WIDTH],
    ] {
        assert!(session.readout_linear_f16(&matrix, &row).is_err());
    }
    assert!(!session.is_poisoned());
    assert_eq!(session.committed_len(), 3);
    assert_eq!(cache_bits(&session), original_cache);
    assert_eq!(bits(&session.append(&[19]).unwrap()), bits(&continuation));
    drop(session);
    drop(matrix);

    let mut session = model.create_session(37).unwrap();
    session.append(&[0]).unwrap();
    let mut bytes = identity_bytes();
    set(&mut bytes, 0, 1, 65504.0);
    let matrix = K2LinearF16::from_target_source_le(bytes).unwrap();
    input[1] = f32::MAX;
    assert!(session.readout_linear_f16(&matrix, &input).is_err());
    assert_eq!(session.committed_len(), 1);
    assert!(session.is_poisoned());
    assert!(matches!(
        session.append(&[42]),
        Err(K2RuntimeError::Poisoned)
    ));
    assert!(matches!(
        session.readout(&capture.residuals),
        Err(K2RuntimeError::Poisoned)
    ));
}
