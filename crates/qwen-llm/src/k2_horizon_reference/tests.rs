use super::*;

#[derive(Deserialize)]
struct Fixture {
    schema: u32,
    geometry: Geometry,
    weights: Weights,
    cases: Vec<Case>,
    corruption: Vec<Corruption>,
}

#[derive(Deserialize)]
struct Case {
    theta: f64,
    base: usize,
    storage: CacheStorage,
    tokens: Vec<u32>,
    traces: Vec<TokenTrace>,
}

#[derive(Deserialize)]
struct Corruption {
    fault: String,
    max_logit_delta: f64,
    logits: Vec<Vec<f32>>,
}

fn fixture() -> Fixture {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../tests/fixtures/k2_math_numpy.json")).unwrap();
    assert_eq!(fixture.schema, 1);
    fixture
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        // F64 reductions differ in order between scalar Rust and NumPy BLAS;
        // retain a small F32 rounding allowance, not a BF16 deployment tolerance.
        let tolerance = 2e-6 + 2e-6 * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{label}[{i}]: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn compare(actual: &TokenTrace, expected: &TokenTrace) {
    assert_eq!(actual.position, expected.position);
    assert_eq!(actual.layers.len(), expected.layers.len());
    for (actual, expected) in actual.layers.iter().zip(&expected.layers) {
        for (a, e, name) in [
            (
                &actual.attention_norm,
                &expected.attention_norm,
                "attention norm",
            ),
            (
                &actual.query_rotated,
                &expected.query_rotated,
                "rotated query",
            ),
            (&actual.key_stored, &expected.key_stored, "stored key"),
            (&actual.value_stored, &expected.value_stored, "stored value"),
            (&actual.attention, &expected.attention, "attention"),
            (
                &actual.post_attention,
                &expected.post_attention,
                "post attention",
            ),
            (
                &actual.feed_forward_norm,
                &expected.feed_forward_norm,
                "FF norm",
            ),
            (&actual.gated, &expected.gated, "SwiGLU"),
            (&actual.residual, &expected.residual, "residual"),
        ] {
            close(a, e, name);
        }
    }
    close(&actual.output_norm, &expected.output_norm, "output norm");
    close(&actual.logits, &expected.logits, "logits");
}

#[test]
fn composed_forward_matches_independent_batch_causal_numpy() {
    let f = fixture();
    assert_eq!(f.cases.len(), 6);
    for (theta, base) in [(500_000.0, 0), (1_000_000.0, 7919), (10_000_000.0, 131069)] {
        for storage in [CacheStorage::F32, CacheStorage::F16] {
            assert_eq!(
                f.cases
                    .iter()
                    .filter(|case| {
                        case.theta == theta && case.base == base && case.storage == storage
                    })
                    .count(),
                1,
                "missing or duplicate positional/storage case"
            );
        }
    }
    assert_ne!(
        f.geometry.hidden,
        f.geometry.query_heads * f.geometry.head_dim
    );
    assert_eq!(f.weights.layers.len(), 2);
    for case in &f.cases {
        let mut geometry = f.geometry.clone();
        geometry.theta = case.theta;
        let model = ReferenceModel::new(geometry, f.weights.clone()).unwrap();
        let mut all = model
            .session(case.storage, case.base, case.tokens.len())
            .unwrap();
        let full = all.append(case.base, &case.tokens).unwrap();
        assert_eq!(full.len(), case.traces.len());
        for (actual, expected) in full.iter().zip(&case.traces) {
            compare(actual, expected);
        }
        // Every split verifies the same absolute positions and rounded cache,
        // not just an all-prefill versus all-single-token special case.
        for split in 1..case.tokens.len() {
            let mut session = model
                .session(case.storage, case.base, case.tokens.len())
                .unwrap();
            let mut trace = session.append(case.base, &case.tokens[..split]).unwrap();
            assert_eq!(session.committed_len(), split);
            trace.extend(
                session
                    .append(case.base + split, &case.tokens[split..])
                    .unwrap(),
            );
            assert_eq!(session.committed_len(), case.tokens.len());
            assert_eq!(trace, full);
            assert_eq!(session.cache, all.cache);
        }
        let mut singles = model
            .session(case.storage, case.base, case.tokens.len())
            .unwrap();
        for (i, &token) in case.tokens.iter().enumerate() {
            assert_eq!(singles.append(case.base + i, &[token]).unwrap()[0], full[i]);
        }
    }
}

#[test]
fn negative_controls_are_distinguishable_at_the_asserted_tolerance() {
    let f = fixture();
    assert_eq!(f.corruption.len(), 7);
    let expected = &f.cases.last().unwrap().traces;
    for corruption in &f.corruption {
        assert_eq!(corruption.logits.len(), expected.len());
        let mut delta = 0.0_f64;
        let mut distinguishable = false;
        for (wrong, correct) in corruption.logits.iter().zip(expected) {
            assert_eq!(wrong.len(), correct.logits.len());
            for (&wrong, &correct) in wrong.iter().zip(&correct.logits) {
                let error = f64::from((wrong - correct).abs());
                delta = delta.max(error);
                distinguishable |= error > f64::from(2e-6 + 2e-6 * correct.abs());
            }
        }
        assert!(distinguishable, "{}", corruption.fault);
        assert!(
            (delta - corruption.max_logit_delta).abs() < 2e-6,
            "{}",
            corruption.fault
        );
    }
}

#[test]
fn invalid_requests_preserve_committed_cache_and_allow_retry() {
    let f = fixture();
    let model = ReferenceModel::new(f.geometry, f.weights).unwrap();
    let mut session = model.session(CacheStorage::F16, 37, 4).unwrap();
    session.append(37, &[2, 7]).unwrap();
    let cache = session.cache.clone();
    for (position, tokens) in [
        (38, vec![3]),
        (39, vec![]),
        (39, vec![23]),
        (39, vec![1, 2, 3]),
    ] {
        assert!(session.append(position, &tokens).is_err());
        assert_eq!(session.committed_len(), 2);
        assert_eq!(session.cache, cache);
    }
    session.append(39, &[3, 11]).unwrap();
    assert_eq!(session.committed_len(), 4);
    assert!(session.append(41, &[2]).is_err());
}

#[test]
fn geometry_weights_and_position_limits_fail_closed() {
    let f = fixture();
    for modify in [
        |g: &mut Geometry| g.hidden = usize::MAX,
        |g: &mut Geometry| g.kv_heads = 0,
        |g: &mut Geometry| g.query_heads = 7,
        |g: &mut Geometry| g.head_dim = 3,
        |g: &mut Geometry| g.norm_groups = 0,
        |g: &mut Geometry| g.theta = f64::NAN,
        |g: &mut Geometry| g.epsilon = 0.0,
    ] {
        let mut g = f.geometry.clone();
        modify(&mut g);
        assert!(ReferenceModel::new(g, f.weights.clone()).is_err());
    }
    let mut wrong = f.weights.clone();
    wrong.layers[0].query.pop();
    assert!(matches!(
        ReferenceModel::new(f.geometry.clone(), wrong),
        Err(ReferenceError::Weight("query"))
    ));
    let mut wrong = f.weights.clone();
    wrong.output[0] = f32::INFINITY;
    assert!(matches!(
        ReferenceModel::new(f.geometry.clone(), wrong),
        Err(ReferenceError::Weight("untied output"))
    ));
    let model = ReferenceModel::new(f.geometry, f.weights).unwrap();
    for (base, capacity) in [(0, 0), (0, 33), (usize::MAX, 2), (524_287, 2)] {
        assert!(model.session(CacheStorage::F32, base, capacity).is_err());
    }
    assert!(model.session(CacheStorage::F32, 524_287, 1).is_ok());
}

#[test]
fn nonfinite_readout_rolls_back_every_layer_and_token() {
    let f = fixture();
    let mut w = f.weights;
    w.embedding.fill(0.0);
    w.embedding[f.geometry.hidden..2 * f.geometry.hidden].fill(1.0);
    w.output.fill(f32::MAX);
    w.output_norm.fill(1.0);
    for layer in &mut w.layers {
        for matrix in [
            &mut layer.query,
            &mut layer.key,
            &mut layer.value,
            &mut layer.attention_output,
            &mut layer.gate,
            &mut layer.up,
            &mut layer.down,
        ] {
            matrix.fill(0.0);
        }
    }
    let model = ReferenceModel::new(f.geometry, w).unwrap();
    for storage in [CacheStorage::F32, CacheStorage::F16] {
        let mut session = model.session(storage, 0, 4).unwrap();
        session.append(0, &[0]).unwrap();
        let committed = session.cache.clone();
        assert_eq!(
            session.append(1, &[0, 1]),
            Err(ReferenceError::Nonfinite("matvec"))
        );
        assert_eq!(session.committed_len(), 1);
        assert_eq!(session.cache, committed);
        session.append(1, &[0]).unwrap();
        assert_eq!(session.committed_len(), 2);
    }
}

#[test]
fn f16_overflow_is_not_silently_stored() {
    assert_eq!(
        CacheStorage::F16.round(&[70_000.0]),
        Err(ReferenceError::Nonfinite("stored K/V"))
    );
    assert_eq!(
        CacheStorage::F32.round(&[70_000.0]).unwrap(),
        vec![70_000.0]
    );
    assert_ne!(CacheStorage::F16.round(&[1.0003]).unwrap()[0], 1.0003);
}

#[test]
fn f16_cache_overflow_rolls_back_staged_rows_and_layers() {
    let f = fixture();
    let mut w = f.weights;
    w.embedding.fill(0.0);
    w.embedding[f.geometry.hidden..2 * f.geometry.hidden].fill(1.0);
    w.layers[1].value.fill(100_000.0);
    let model = ReferenceModel::new(f.geometry, w).unwrap();
    let mut session = model.session(CacheStorage::F16, 0, 4).unwrap();
    session.append(0, &[0]).unwrap();
    let committed = session.cache.clone();
    assert_eq!(
        session.append(1, &[0, 1]),
        Err(ReferenceError::Nonfinite("stored K/V"))
    );
    assert_eq!(session.committed_len(), 1);
    assert_eq!(session.cache, committed);
    session.append(1, &[0]).unwrap();
}
