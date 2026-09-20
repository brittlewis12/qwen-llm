use super::*;

fn bytes(session: &K2Session<'_, '_>) -> Vec<u8> {
    let cache = &session.buffers.cache;
    assert_eq!(cache.offset, 0);
    unsafe {
        std::slice::from_raw_parts(
            cache.buffer.contents().as_ptr().cast::<u8>(),
            cache.n_bytes() as usize,
        )
        .to_vec()
    }
}

fn poison_empty(session: &K2Session<'_, '_>) {
    assert_eq!(session.committed_len(), 0);
    unsafe {
        std::slice::from_raw_parts_mut(
            session
                .buffers
                .cache
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>(),
            session.buffers.cache.n_bytes() as usize,
        )
        .fill(0xff);
    }
}

fn bitwise(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[test]
#[ignore = "GPU experimental compact KV invariants; production lease and K2_GGUF; not quality qualification"]
fn gpu_compact_cache_preserves_transactions_causality_and_lens() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(
        K2LoadedModel::load_with_storage_unqualified(
            &ctx,
            &source,
            4,
            AttentionBackend::Materialized,
            K2KvStorage::Q8_0
        )
        .is_err()
    );
    let mut model = K2LoadedModel::load_with_storage_unqualified(
        &ctx,
        &source,
        4,
        AttentionBackend::Online,
        K2KvStorage::Q8_0,
    )
    .unwrap();
    assert_eq!(model.prefill, PrefillMode::Serial);
    let mut serial = model.create_session(37).unwrap();
    assert_eq!(serial.buffers.cache.dtype, GgmlType::I8);
    assert_eq!(serial.buffers.cache.n_bytes(), 4 * 78336);
    assert_eq!(
        model.plan.session.buffer_bytes().iter().sum::<u64>(),
        4 * 78336 + 1_240_068
    );
    poison_empty(&serial);
    let empty = bytes(&serial);
    serial.buffers.cache.dtype = GgmlType::F16;
    let error = serial.append(&[0]).unwrap_err();
    assert!(error.to_string().contains("cache storage/shape"));
    serial.buffers.cache.dtype = GgmlType::I8;
    assert_eq!(serial.committed_len(), 0);
    assert!(!serial.is_poisoned());
    assert!(bytes(&serial) == empty);
    assert!(serial.append(&[0, 42, 250624]).is_err());
    assert!(bytes(&serial) == empty);
    let sites = [0, 11, 23, 35];
    let expected = serial.append_with_captures(&[0, 42, 17], &sites).unwrap();
    let prefix = bytes(&serial);
    for layer in 0..36 {
        let ranges = serial.request.layer_planes(layer).unwrap();
        for range in [ranges.key, ranges.value] {
            assert!(
                prefix[(range.end - 1088) as usize..range.end as usize]
                    .iter()
                    .all(|&b| b == 255)
            );
        }
    }
    assert!(bitwise(
        &serial.readout(&expected.residuals[3 * 4096..]).unwrap(),
        &expected.logits
    ));
    assert!(bytes(&serial) == prefix);
    let next = serial.append(&[19]).unwrap();
    let full = bytes(&serial);
    assert!(serial.append(&[20]).is_err());
    assert_eq!(serial.committed_len(), 4);
    assert!(!serial.is_poisoned());
    assert!(bytes(&serial) == full);
    drop(serial);

    // Exercise singleton Q8 first; only then enable packed scheduling on that layout.
    model.prefill = PrefillMode::BatchQ8;
    let mut packed = model.create_session(37).unwrap();
    poison_empty(&packed);
    let actual = packed.append_with_captures(&[0, 42, 17], &sites).unwrap();
    assert!(bitwise(&actual.logits, &expected.logits));
    assert!(bitwise(&actual.residuals, &expected.residuals));
    assert!(bytes(&packed) == prefix);
    assert!(bitwise(
        read_f32(&packed.buffers.residual),
        &actual.residuals[3 * 4096..]
    ));
    assert!(bitwise(&packed.append(&[19]).unwrap(), &next));
    assert!(bytes(&packed) == full);
    drop(packed);

    let direction = vec![0.0001; 4096];
    let operations = [
        K2Intervention {
            post_block_layer: 0,
            coefficient: 0.25,
            kind: K2InterventionKind::Fixed {
                direction: &direction,
            },
        },
        K2Intervention {
            post_block_layer: 35,
            coefficient: -0.125,
            kind: K2InterventionKind::Fixed {
                direction: &direction,
            },
        },
    ];
    model.prefill = PrefillMode::Serial;
    let mut control = model.create_session(37).unwrap();
    let expected = control
        .append_with_interventions(&[0, 42, 17], &sites, &operations)
        .unwrap();
    let expected_cache = bytes(&control);
    let continuation = control.append(&[19]).unwrap();
    let continued_cache = bytes(&control);
    drop(control);
    model.prefill = PrefillMode::BatchQ8;
    let mut candidate = model.create_session(37).unwrap();
    let actual = candidate
        .append_with_interventions(&[0, 42, 17], &sites, &operations)
        .unwrap();
    assert!(
        bitwise(&actual.logits, &expected.logits)
            && bitwise(&actual.residuals, &expected.residuals)
    );
    assert!(bytes(&candidate) == expected_cache);
    assert!(bitwise(&candidate.append(&[19]).unwrap(), &continuation));
    assert!(bytes(&candidate) == continued_cache);
    drop(candidate);

    let mut failed = model.create_session(37).unwrap();
    let huge = vec![f32::MAX; 4096];
    let error = failed
        .append_with_interventions(
            &[0, 42, 17],
            &sites,
            &[K2Intervention {
                post_block_layer: 0,
                coefficient: f32::MAX,
                kind: K2InterventionKind::Fixed { direction: &huge },
            }],
        )
        .unwrap_err();
    assert!(error.to_string().contains("nonfinite"));
    assert!(
        bytes(&failed)
            .chunks_exact(34)
            .any(|b| u16::from_le_bytes([b[0], b[1]]) == 0x7e00)
    );
    assert_eq!(failed.committed_len(), 0);
    assert!(failed.is_poisoned());
    assert!(matches!(failed.append(&[0]), Err(K2RuntimeError::Poisoned)));
    drop(failed);
    let mut fresh = model.create_session(37).unwrap();
    assert!(
        fresh
            .append(&[0, 42])
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}
