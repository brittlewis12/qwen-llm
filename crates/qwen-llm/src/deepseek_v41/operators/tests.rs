use super::*;

fn matrix(n: usize) -> Linear {
    Linear {
        input: n,
        output: n,
        weight: (0..n * n)
            .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
            .collect(),
    }
}
fn weights(layer: usize) -> AttentionWeights {
    let owns = [2, 8, 14, 20].contains(&layer);
    AttentionWeights {
        query_a: matrix(2),
        query_norm: vec![1.0; 2],
        query_b: matrix(2),
        kv: matrix(2),
        kv_norm: vec![1.0; 2],
        sink: vec![0.0],
        output_a: vec![matrix(2)],
        output_b: matrix(2),
        compressor: owns.then(|| CompressorWeights {
            ratio: if layer < 20 { 2 } else { 1 },
            kv: matrix(2),
            gate: Some(Linear {
                input: 2,
                output: 2,
                weight: vec![0.0; 4],
            }),
            norm: vec![1.0; 2],
        }),
        index: [2, 8, 14, 20, 24, 28, 32, 36]
            .contains(&layer)
            .then(|| IndexWeights {
                query: matrix(2),
                weights: Linear {
                    input: 2,
                    output: 1,
                    weight: vec![1.0, 0.0],
                },
                key: owns.then(|| matrix(2)),
                key_norm: vec![1.0; 2],
                heads: 1,
                dim: 2,
                topk: 2,
            }),
        frequencies: vec![0.25],
        head_dim: 2,
        window: 2,
        eps: 1e-6,
        candidate_block_size: 2,
        candidate_topk_blocks: 1,
    }
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}

#[test]
fn compression_pairs_and_continuation() {
    let mut state = CompressorState::default();
    let mut w = weights(2).compressor.unwrap();
    w.gate = Some(matrix(2));
    assert!(state.push(&w, &[1.0, 3.0], 0, 1e-6).unwrap().is_none());
    let pooled = state.push(&w, &[3.0, 1.0], 1, 1e-6).unwrap().unwrap();
    close(&pooled, &[1.0, 1.0]);
    assert!(state.push(&w, &[2.0, 0.0], 2, 1e-6).unwrap().is_none());
    let mut resumed = state.clone();
    close(
        &state.push(&w, &[0.0, 2.0], 3, 1e-6).unwrap().unwrap(),
        &resumed.push(&w, &[0.0, 2.0], 3, 1e-6).unwrap().unwrap(),
    );
    assert!(state.push(&w, &[0.0, 2.0], 3, 1e-6).is_err());
}

#[test]
fn all_layers_all_pair_phases_chunking_and_sessions() {
    let layers: Vec<_> = (0..40).map(|l| (l, weights(l))).collect();
    let rows: Vec<_> = (0..7).map(|i| vec![1.0 + i as f32 * 0.1, 0.7]).collect();
    let expected = Session::default().decode_rows(0, &rows, &layers).unwrap();
    for cut in 1..7 {
        let mut session = Session::default();
        let mut actual = session.decode_rows(0, &rows[..cut], &layers).unwrap();
        let mut resumed = session.clone();
        actual.extend(resumed.decode_rows(cut, &rows[cut..], &layers).unwrap());
        for (a, b) in actual.iter().zip(&expected) {
            close(a, b);
        }
    }
    let mut a = Session::default();
    let mut b = Session::default();
    for i in 0..rows.len() {
        close(
            &a.decode_rows(i, &rows[i..i + 1], &layers).unwrap()[0],
            &expected[i],
        );
        close(
            &b.decode_rows(i, &rows[i..i + 1], &layers).unwrap()[0],
            &expected[i],
        );
    }
}

#[test]
fn incomplete_pair_republishes_own_keys_and_stale_owner_rejected() {
    let mut session = Session::default();
    for pos in 0..3 {
        for layer in [2, 8, 14, 20] {
            let mut w = weights(layer);
            w.kv_norm = vec![1.0 + layer as f32, 1.0];
            w.compressor.as_mut().unwrap().norm = w.kv_norm.clone();
            session.attention(layer, pos, &[1.0, 0.5], &w).unwrap();
            assert_eq!(session.published, Some((layer, pos)));
        }
    }
    // Owner 20 was last published. A layer-3 read must not see it.
    assert!(
        session
            .attention(3, 0, &[1.0, 0.5], &weights(3))
            .unwrap_err()
            .contains("stale")
    );
    assert!(!session.layers.contains_key(&3));
}

#[test]
fn sparse_sink_rotary_and_group_projection() {
    close(
        &sparse_attention(&[0.0, 0.0], &[vec![2.0, 4.0]], &[0], 0.0).unwrap(),
        &[1.0, 2.0],
    );
    let mut x = vec![7.0, 2.0, 3.0];
    let original = x.clone();
    rotary(&mut x, &[0.7], 13, false).unwrap();
    rotary(&mut x, &[0.7], 13, true).unwrap();
    close(&x, &original);
    let mut w = weights(0);
    w.frequencies.clear();
    let input = vec![1.0, 2.0];
    let q = rms_norm(&input, &[1.0; 2], w.eps).unwrap();
    let expected = sparse_attention(&q, &[q.clone()], &[0], 0.0).unwrap();
    close(
        &Session::default().attention(0, 0, &input, &w).unwrap(),
        &expected,
    );
    w.output_a = vec![
        Linear {
            input: 1,
            output: 1,
            weight: vec![2.0],
        },
        Linear {
            input: 1,
            output: 1,
            weight: vec![3.0],
        },
    ];
    close(
        &Session::default().attention(0, 0, &input, &w).unwrap(),
        &[expected[0] * 2.0, expected[1] * 3.0],
    );
}

#[test]
fn candidates_pin_recent_and_topk_is_position_sorted() {
    close(
        &rotary_frequencies(4, 10000.0, 0, 40.0, 32.0, 1.0).unwrap(),
        &[1.0, 0.01],
    );
    assert_eq!(
        candidate_blocks(&[99.0, 98.0, 1.0, 2.0, 0.0], 2, 1).unwrap(),
        vec![false, false, false, false, true]
    );
    assert_eq!(topk(&[3.0, 5.0, 4.0], 2), vec![1, 2]);
}

#[test]
fn assembled_attention_ffn_and_final_collapse() {
    let hc = HcWeights {
        projection: Linear {
            input: 4,
            output: 8,
            weight: vec![0.0; 32],
        },
        scale: [1.0; 3],
        base: vec![0.0; 8],
        streams: 2,
        iterations: 20,
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    };
    let mut session = Session::default();
    for position in 0..5 {
        let mut residual = vec![1.0, 0.5, 1.0, 0.5];
        let mut pre = vec![1.0, 0.0];
        for layer in 0..40 {
            (residual, pre) = lagged_sublayer(&residual, &pre, &hc, &[1.0; 2], |x| {
                session.attention(layer, position, x, &weights(layer))
            })
            .unwrap();
            (residual, pre) =
                lagged_sublayer(&residual, &pre, &hc, &[1.0; 2], |x| matrix(2).forward(x)).unwrap();
        }
        let logits = matrix(2)
            .forward(&rms_norm(&collapse(&residual, &pre).unwrap(), &[1.0; 2], 1e-6).unwrap())
            .unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn lagged_mhc_timing_and_source_major_merge() {
    let weights = HcWeights {
        projection: Linear {
            input: 4,
            output: 8,
            weight: vec![0.0; 32],
        },
        scale: [1.0; 3],
        base: vec![0.0; 8],
        streams: 2,
        iterations: 20,
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    };
    let residual = [1.0, 2.0, 8.0, 4.0];
    let (after, pre) = lagged_sublayer(&residual, &[1.0, 0.0], &weights, &[1.0; 2], |input| {
        close(input, &rms_norm(&residual[..2], &[1.0; 2], 1e-6).unwrap());
        Ok(input.to_vec())
    })
    .unwrap();
    let (_, next) = lagged_sublayer(&after, &pre, &weights, &[1.0; 2], |input| {
        close(
            input,
            &rms_norm(&collapse(&after, &pre).unwrap(), &[1.0; 2], 1e-6).unwrap(),
        );
        Ok(input.to_vec())
    })
    .unwrap();
    close(&next, &[0.500001; 2]);
    close(
        &merge(
            &[0.0],
            &[2.0, 7.0],
            &Mixes {
                pre: vec![],
                post: vec![0.0; 2],
                comb: vec![1.0, 2.0, 3.0, 4.0],
            },
        )
        .unwrap(),
        &[23.0, 32.0],
    );
    let mixes = weights.mixes(&residual).unwrap();
    close(&mixes.comb, &[0.5; 4]);
}
