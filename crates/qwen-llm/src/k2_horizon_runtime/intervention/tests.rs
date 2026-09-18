use super::*;

fn fixed(layer: u32, coefficient: f32, direction: &[f32]) -> K2Intervention<'_> {
    K2Intervention {
        post_block_layer: layer,
        coefficient,
        kind: K2InterventionKind::Fixed { direction },
    }
}

#[test]
fn host_validation_bounds_all_vectors_and_preserves_same_site_order() {
    let row = vec![0.25; WIDTH];
    assert_eq!(validate(&[]).unwrap(), 0);
    let operations = [
        fixed(0, 1.0, &row),
        K2Intervention {
            post_block_layer: 0,
            coefficient: -1.0,
            kind: K2InterventionKind::ResidualL2Relative { direction: &row },
        },
        K2Intervention {
            post_block_layer: 35,
            coefficient: 0.5,
            kind: K2InterventionKind::Projection { direction: &row },
        },
        K2Intervention {
            post_block_layer: 35,
            coefficient: 0.5,
            kind: K2InterventionKind::SourceToTarget {
                source: &row,
                target: &row,
            },
        },
    ];
    assert_eq!(validate(&operations).unwrap(), 5);
    assert!(matches!(
        operations[0].kind,
        K2InterventionKind::Fixed { .. }
    ));
    assert!(matches!(
        operations[1].kind,
        K2InterventionKind::ResidualL2Relative { .. }
    ));
    assert!(validate(&[fixed(35, 1.0, &row), fixed(0, 1.0, &row)]).is_err());
    assert_eq!(
        validate(&vec![operations[3]; 64]).unwrap() * WIDTH * 4,
        2 * 1024 * 1024
    );
    assert!(validate(&vec![operations[3]; 65]).is_err());
    for layer in [36, u32::MAX] {
        assert!(validate(&[fixed(layer, 1.0, &row)]).is_err());
    }
    for coefficient in [0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(validate(&[fixed(0, coefficient, &row)]).is_err());
    }
    for invalid_row in [
        vec![],
        vec![0.0; WIDTH - 1],
        vec![0.0; WIDTH + 1],
        vec![f32::NAN; WIDTH],
        vec![f32::INFINITY; WIDTH],
    ] {
        assert!(validate(&[fixed(0, 1.0, &invalid_row)]).is_err());
        for (source, target) in [(&invalid_row[..], &row[..]), (&row[..], &invalid_row[..])] {
            assert!(
                validate(&[K2Intervention {
                    post_block_layer: 0,
                    coefficient: 1.0,
                    kind: K2InterventionKind::SourceToTarget { source, target }
                }])
                .is_err()
            );
        }
    }
    // Finite does not mean overflow-proof; post-submit checks own that failure.
    assert!(validate(&[fixed(35, f32::MAX, &vec![f32::MAX; WIDTH])]).is_ok());
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn cache_bits(session: &K2Session<'_, '_>) -> Vec<u16> {
    // Private checked shared F16 arena, read only between synchronous commands.
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

fn scalar(x: &[f32], operation: K2Intervention<'_>) -> Vec<f32> {
    let x = x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let (source, target) = operation.kind.vectors();
    let coefficient = f64::from(operation.coefficient);
    let dot = x
        .iter()
        .zip(source)
        .map(|(x, &v)| x * f64::from(v))
        .sum::<f64>();
    let l2 = x.iter().map(|x| x * x).sum::<f64>().sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &x)| {
            let v = f64::from(source[i]);
            (match operation.kind {
                K2InterventionKind::Fixed { .. } => x + coefficient * v,
                K2InterventionKind::ResidualL2Relative { .. } => x + coefficient * l2 * v,
                K2InterventionKind::Projection { .. } => x - coefficient * dot * v,
                K2InterventionKind::SourceToTarget { .. } => {
                    x + coefficient * dot * (f64::from(target.unwrap()[i]) - v)
                }
            }) as f32
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < 2e-5 * (1.0 + b.abs()),
            "coordinate {i}: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "full checkpoint GPU intervention correctness; production lease and K2_GGUF required"]
fn gpu_ordered_interventions_match_formulas_and_preserve_causal_kv() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 4).unwrap();
    let mut plain = model.create_session(37).unwrap();
    let baseline = plain
        .append_with_captures(&[0, 42, 17], &[0, 1, 35])
        .unwrap();
    let baseline_cache = cache_bits(&plain);
    let baseline_continuation = plain.append(&[19]).unwrap();
    drop(plain);

    let mut noop = model.create_session(37).unwrap();
    let noop_output = noop
        .append_with_interventions(&[0, 42, 17], &[], &[])
        .unwrap();
    assert!(noop_output.residuals.is_empty() && noop_output.post_block_layers.is_empty());
    assert_eq!(bits(&noop_output.logits), bits(&baseline.logits));
    assert_eq!(cache_bits(&noop), baseline_cache);
    let row = vec![0.0; WIDTH];
    assert!(
        noop.append_with_interventions(&[19], &[], &[fixed(36, 1.0, &row)])
            .is_err()
    );
    assert!(
        noop.append_with_interventions(&[19], &[35, 0], &[])
            .is_err()
    );
    assert_eq!(noop.committed_len(), 3);
    assert!(!noop.is_poisoned());
    assert_eq!(
        bits(&noop.append(&[19]).unwrap()),
        bits(&baseline_continuation)
    );
    drop(noop);

    let direction = (0..WIDTH)
        .map(|i| ((i % 17) as f32 - 8.0) / 512.0)
        .collect::<Vec<_>>();
    let target = (0..WIDTH)
        .map(|i| ((i % 13) as f32 - 6.0) / 256.0)
        .collect::<Vec<_>>();
    for kind in [
        K2InterventionKind::Fixed {
            direction: &direction,
        },
        K2InterventionKind::ResidualL2Relative {
            direction: &direction,
        },
        K2InterventionKind::Projection {
            direction: &direction,
        },
        K2InterventionKind::SourceToTarget {
            source: &direction,
            target: &target,
        },
    ] {
        let operation = K2Intervention {
            post_block_layer: 35,
            coefficient: 0.375,
            kind,
        };
        let mut session = model.create_session(37).unwrap();
        let output = session
            .append_with_interventions(&[0, 42, 17], &[35], &[operation])
            .unwrap();
        assert_close(
            &output.residuals,
            &scalar(&baseline.residuals[2 * WIDTH..], operation),
        );
        assert_ne!(bits(&output.logits), bits(&baseline.logits));
        assert_eq!(
            bits(&session.readout(&output.residuals).unwrap()),
            bits(&output.logits)
        );
        assert_eq!(cache_bits(&session), baseline_cache);
        assert_eq!(
            bits(&session.append(&[19]).unwrap()),
            bits(&baseline_continuation)
        );
    }

    let mut basis = vec![0.0; WIDTH];
    basis[0] = 1.0;
    let add = fixed(35, 0.75, &basis);
    let remove = K2Intervention {
        post_block_layer: 35,
        coefficient: 1.0,
        kind: K2InterventionKind::Projection { direction: &basis },
    };
    let mut results = Vec::new();
    for operations in [[add, remove], [remove, add]] {
        let mut session = model.create_session(37).unwrap();
        let output = session
            .append_with_interventions(&[0, 42, 17], &[35], &operations)
            .unwrap();
        let mut expected = baseline.residuals[2 * WIDTH..].to_vec();
        for operation in operations {
            expected = scalar(&expected, operation);
        }
        assert_close(&output.residuals, &expected);
        results.push(output.residuals);
    }
    assert_eq!(results[0][0], 0.0);
    assert_eq!(results[1][0], 0.75);
    assert_ne!(bits(&results[0]), bits(&results[1]));

    let early = fixed(0, 0.75, &basis);
    let mut session = model.create_session(37).unwrap();
    let output = session
        .append_with_interventions(&[0, 42, 17], &[0, 1, 35], &[early])
        .unwrap();
    assert_eq!(output.absolute_position, 39);
    assert_close(
        &output.residuals[..WIDTH],
        &scalar(&baseline.residuals[..WIDTH], early),
    );
    let changed_cache = cache_bits(&session);
    let plan = session.request.append(0, 37, 3).unwrap();
    let last = plan.token(2).unwrap();
    let mut allowed = baseline_cache.clone();
    for layer in 1..36 {
        let ranges = last.write_ranges(layer).unwrap();
        let mut changed = false;
        for range in [ranges.key, ranges.value] {
            let start = (range.start / 2) as usize;
            let end = (range.end / 2) as usize;
            changed |= changed_cache[start..end] != baseline_cache[start..end];
            allowed[start..end].copy_from_slice(&changed_cache[start..end]);
        }
        assert!(changed, "expected later layer {layer} KV to change");
    }
    assert_eq!(
        changed_cache, allowed,
        "only later-layer current-token KV may change"
    );
    let changed_continuation = session.append(&[19]).unwrap();
    assert_ne!(bits(&changed_continuation), bits(&baseline_continuation));
    drop(session);

    let mut split = model.create_session(37).unwrap();
    split.append(&[0, 42]).unwrap();
    let split_output = split
        .append_with_interventions(&[17], &[0, 1, 35], &[early])
        .unwrap();
    assert_eq!(bits(&split_output.residuals), bits(&output.residuals));
    assert_eq!(bits(&split_output.logits), bits(&output.logits));
    assert_eq!(cache_bits(&split), changed_cache);
    assert_eq!(
        bits(&split.append(&[19]).unwrap()),
        bits(&changed_continuation)
    );
    drop(split);

    let mut overflow = model.create_session(37).unwrap();
    let huge = vec![f32::MAX; WIDTH];
    assert!(
        overflow
            .append_with_interventions(&[0, 42, 17], &[], &[fixed(35, f32::MAX, &huge)])
            .is_err()
    );
    assert_eq!(overflow.committed_len(), 0);
    assert!(overflow.is_poisoned());
    assert!(matches!(
        overflow.append(&[19]),
        Err(K2RuntimeError::Poisoned)
    ));
    assert!(matches!(
        overflow.readout(&basis),
        Err(K2RuntimeError::Poisoned)
    ));
}
