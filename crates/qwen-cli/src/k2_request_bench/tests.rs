use super::*;
use std::cell::RefCell;

fn parse(extra: &[&str]) -> Result<K2RequestArgs, clap::Error> {
    K2RequestArgs::try_parse_from(
        ["k2-request", "--model", "unused.gguf"]
            .into_iter()
            .chain(extra.iter().copied()),
    )
}

#[test]
fn input_and_budget_flags_are_explicit_and_exclusive() {
    assert!(parse(&["--raw-prompt", "text", "--tokens", "1"]).is_ok());
    assert!(
        parse(&[
            "--raw-prompt",
            "<|ifm|begin_of_text|>text",
            "--no-special-tokens",
            "--tokens",
            "1"
        ])
        .unwrap()
        .no_special_tokens
    );
    assert!(parse(&["--token-ids", "0,42", "--tokens", "1"]).is_ok());
    for flags in [
        vec!["--raw-prompt", "text"],
        vec!["--tokens", "1"],
        vec!["--raw-prompt", "text", "--token-ids", "0", "--tokens", "1"],
        vec!["--token-ids", "0", "--no-special-tokens", "--tokens", "1"],
        vec![
            "--raw-prompt",
            "text",
            "--tokens",
            "1",
            "--messages",
            "unused",
        ],
        vec![
            "--raw-prompt",
            "text",
            "--tokens",
            "1",
            "--temperature",
            "0.5",
        ],
    ] {
        assert!(parse(&flags).is_err(), "{flags:?}");
    }
}

#[test]
fn budgets_ids_and_stops_reject_before_runtime() {
    assert_eq!(budget(6, 8, None, 524288).unwrap(), 13);
    assert_eq!(budget(32, 1, None, 32).unwrap(), 32);
    assert_eq!(budget(1, 32, Some(32), 32).unwrap(), 32);
    assert_eq!(budget(256, 1, None, 8192).unwrap(), 256);
    assert_eq!(budget(1, 256, Some(256), 8192).unwrap(), 256);
    for size in [257, 1024, 7169, 8192, 524288] {
        assert_eq!(budget(size, 1, None, 524288).unwrap(), size);
        assert_eq!(budget(1, size, None, 524288).unwrap(), size);
    }
    for (prompt, tokens, capacity, context) in [
        (0, 1, None, 32),
        (1, 0, None, 32),
        (1, 33, None, 32),
        (32, 2, None, 32),
        (1, 1, Some(524289), 524288),
        (524289, 1, None, 524288),
        (524288, 2, None, 524288),
        (1, 524289, None, 524288),
        (3, 2, Some(3), 32),
        (3, 2, None, 3),
        (usize::MAX, 2, None, 32),
    ] {
        assert!(budget(prompt, tokens, capacity, context).is_err());
    }
    for id in [-1, 250624, i32::MAX] {
        assert!(checked_id(id, 250624).is_err());
    }
    assert_eq!(checked_id(250623, 250624).unwrap(), 250623);
    assert!(stops(&[1]).is_ok());
    for ids in [vec![], vec![0], vec![1, 2], vec![2, 1]] {
        assert!(stops(&ids).is_err());
    }
}

fn mocked(ids: &[i32], limit: usize) -> (Generation, Vec<String>) {
    let log = RefCell::new(Vec::new());
    let mut cursor = ids.iter();
    let started = Instant::now();
    let generation = generate(
        vec![0.0],
        limit,
        10,
        started,
        |_| {
            let id = *cursor.next().unwrap();
            log.borrow_mut().push(format!("select:{id}"));
            Ok(id)
        },
        |id| {
            log.borrow_mut().push(format!("forward:{id}"));
            Ok(vec![0.0])
        },
        |id| {
            log.borrow_mut().push(format!("emit:{id}"));
            Ok(())
        },
        || Ok(()),
    )
    .unwrap();
    assert!(generation.first_sample_ready_request_ns <= elapsed(started));
    assert!(generation.transition_wall_ns <= generation.wall_ns);
    (generation, log.into_inner())
}

#[test]
fn eos_and_budget_terminal_samples_are_never_forwarded() {
    let (first, log) = mocked(&[1], 8);
    assert_eq!(log, ["select:1"]);
    assert_eq!(first.sampled, [1]);
    assert!(first.emitted.is_empty());
    assert_eq!(first.transition_forwards, 0);
    assert_eq!(first.termination, "eos");
    let (late, log) = mocked(&[3, 4, 1], 8);
    assert_eq!(
        log,
        [
            "select:3",
            "emit:3",
            "forward:3",
            "select:4",
            "emit:4",
            "forward:4",
            "select:1"
        ]
    );
    assert_eq!(late.sampled, [3, 4, 1]);
    assert_eq!(late.emitted, [3, 4]);
    assert_eq!(late.transition_forwards, 2);
    let (budget, log) = mocked(&[3, 4, 5], 3);
    assert_eq!(log.last().unwrap(), "emit:5");
    assert_eq!(budget.sampled, [3, 4, 5]);
    assert_eq!(budget.emitted, [3, 4, 5]);
    assert_eq!(budget.transition_forwards, 2);
    assert_eq!(budget.termination, "token_limit");
    let (one, log) = mocked(&[3], 1);
    assert_eq!(log, ["select:3", "emit:3"]);
    assert_eq!(one.transition_forwards, 0);
}

#[test]
fn invalid_generated_ids_and_callback_failures_never_forward_extra_work() {
    for token in [-1, 10] {
        assert!(
            generate(
                vec![0.0],
                2,
                10,
                Instant::now(),
                |_| Ok(token),
                |_| panic!("invalid token forwarded"),
                |_| panic!("invalid token emitted"),
                || Ok(())
            )
            .is_err()
        );
    }
    assert!(
        generate(
            vec![0.0],
            2,
            10,
            Instant::now(),
            |_| Ok(3),
            |_| panic!("forward after emit failure"),
            |_| anyhow::bail!("sink failed"),
            || Ok(())
        )
        .is_err()
    );
    assert!(
        generate(
            vec![0.0],
            2,
            10,
            Instant::now(),
            |_| panic!("select after cancellation"),
            |_| panic!("forward after cancellation"),
            |_| Ok(()),
            || anyhow::bail!("cancelled")
        )
        .is_err()
    );
}

fn sample(token: i32) -> Sample {
    Sample {
        repetition: 1,
        session_allocation_wall_ns: 1,
        sampler_setup_wall_ns: 1,
        prefill_wall_ns: 10,
        generation_wall_ns: 10,
        transition_forward_wall_ns: 0,
        first_sample_ready_request_wall_ns: 15,
        request_wall_ns: 25,
        prompt_forwards: 2,
        committed_positions: 2,
        observed_session_metal_allocation_delta_bytes: 1024,
        outcome: Outcome {
            sampled_token_ids: vec![token],
            emitted_token_ids: vec![token],
            sampled_token_ids_sha256_i32le: token_ids_sha256_i32le(&[token]),
            emitted_bytes_sha256: format!("{:x}", Sha256::digest(b"raw")),
            emitted_bytes_hex: "726177".into(),
            emitted_text_lossy: "raw".into(),
            emitted_utf8_valid: true,
            termination: "token_limit",
            transition_forwards: 0,
        },
    }
}

#[test]
fn zero_transition_rates_are_null_and_inconsistency_suppresses_aggregation() {
    assert_eq!(rate(0, 1), None);
    assert_eq!(rate(1, 0), None);
    assert_eq!(rate(2, 1_000_000_000), Some(2.0));
    let (consistent, aggregate) = aggregate(Some(&sample(3)), &[sample(3), sample(3)]);
    assert!(consistent);
    let aggregate = aggregate.unwrap();
    assert!(aggregate["transition_forwards_per_second"].is_null());
    assert_eq!(aggregate["mean_request_wall_ns"], 25.0);
    assert_eq!(
        super::aggregate(Some(&sample(4)), &[sample(3)]),
        (false, None)
    );
    assert_eq!(
        super::aggregate(None, &[sample(3), sample(4)]),
        (false, None)
    );
    assert_eq!(super::aggregate(None, &[]), (false, None));
    let json = serde_json::to_value(sample(3)).unwrap();
    assert_eq!(
        json["outcome"]["sampled_token_ids_sha256_i32le"],
        token_ids_sha256_i32le(&[3])
    );
    assert_eq!(
        json["outcome"]["emitted_bytes_sha256"],
        format!("{:x}", Sha256::digest(b"raw"))
    );
}

#[test]
fn each_fresh_greedy_sampler_has_identical_selection_policy() {
    let logits = [0.0, -1.0, 3.0, 3.0];
    let tokens = (0..4)
        .map(|_| {
            Sampler::new(SamplingConfig::default())
                .unwrap()
                .sample(&logits)
                .unwrap()
                .token
        })
        .collect::<Vec<_>>();
    assert!(tokens.iter().all(|&token| token == tokens[0]));
}
