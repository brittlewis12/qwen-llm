use super::*;

fn logits(token: usize) -> Vec<f32> {
    let mut logits = vec![-10.0; 4];
    logits[token] = 10.0;
    logits
}

#[test]
fn ordinary_decode_adapter_preserves_indices_stop_names_and_unconsumed_terminal() {
    for (limit, stops, expected, reason) in [
        (1, vec![], vec![1], "max_new_tokens"),
        (1, vec![1], vec![1], "stop_token"),
        (3, vec![], vec![1, 2, 3], "max_new_tokens"),
        (3, vec![3], vec![1, 2, 3], "stop_token"),
        (3, vec![2], vec![1, 2], "stop_token"),
    ] {
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let mut events = Vec::new();
        let (tokens, stop) = decode_ordinary_tokens(
            logits(1),
            limit,
            &stops.into_iter().collect(),
            &mut sampler,
            |index, token| {
                let phase = Phase::Decode(index);
                events.push((phase.label(), phase.index(), 7 + index, token));
                Ok(logits(token as usize + 1))
            },
        )
        .unwrap();
        assert_eq!(tokens, expected);
        assert_eq!(stop, reason);
        let expected_events = expected[..expected.len() - 1]
            .iter()
            .enumerate()
            .map(|(index, token)| ("decode", index, 7 + index, *token))
            .collect::<Vec<_>>();
        assert_eq!(events, expected_events);
        assert_eq!(sampler.draws(), 0);
    }
}

#[test]
fn ordinary_decode_adapter_keeps_seeded_sampler_progress_and_forward_errors() {
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0x1234_5678_9abc_def0,
    };
    let mut sampler = Sampler::new(config).unwrap();
    let mut consumed = Vec::new();
    let (tokens, reason) = decode_ordinary_tokens(
        vec![2.0, 1.5, 1.0, 0.5],
        4,
        &HashSet::new(),
        &mut sampler,
        |index, token| {
            consumed.push((index, token));
            Ok(vec![2.0, 1.5, 1.0, 0.5])
        },
    )
    .unwrap();
    assert_eq!(tokens, [0, 1, 1, 1]);
    assert_eq!(reason, "max_new_tokens");
    assert_eq!(consumed, [(0, 0), (1, 1), (2, 1)]);
    assert_eq!(sampler.draws(), 4);

    let mut sampler = Sampler::new(config).unwrap();
    let mut calls = 0;
    let error = decode_ordinary_tokens(
        logits(1),
        3,
        &HashSet::new(),
        &mut sampler,
        |index, token| {
            calls += 1;
            assert_eq!((index, token), (0, 1));
            bail!("Lens forward failed")
        },
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "Lens forward failed");
    assert_eq!(calls, 1);
    assert_eq!(sampler.draws(), 1);
}
