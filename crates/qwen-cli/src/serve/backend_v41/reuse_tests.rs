use super::super::{decode_loop, http::PromptBoundaries, snapshot_cache::SnapshotCache};
use super::*;
use serde_json::{Value, json};

fn tokenizer() -> NativeTokenizer {
    tokenizer_fixture::tokenizer(None)
}

fn render(input: Value, thinking: bool, tools: bool) -> (String, PromptBoundaries) {
    let profile = super::super::protocol_v41::profile(OutputLimits::TEST);
    let mut body = json!({"model":"fixture", "input":input});
    if thinking {
        body["reasoning"] = json!({"effort":"high"});
    }
    if tools {
        body["tools"] = json!([{"type":"function","name":"lookup","parameters":{"type":"object"}}]);
    }
    let request = profile.parse(&body).unwrap();
    let (prompt, boundaries) = profile.render_prepared(&request).unwrap();
    (prompt, boundaries.unwrap())
}

#[test]
fn official_text_reasoning_and_tool_rerenders_choose_exact_prefixes() {
    let tokenizer = tokenizer();
    let encode = |text: &str| {
        tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>()
    };
    // Native JoyAI fixture exercises actual recipe rerendering, including history
    // reasoning omission and tool-call canonicalization; no append-only prompt.
    for (thinking, tools, output, turns, live_expected) in [
        (
            false,
            false,
            "answer",
            json!([
                {"role":"user","content":"question"},
                {"role":"assistant","content":"answer"},
                {"role":"user","content":"and next?"}
            ]),
            true,
        ),
        (
            true,
            false,
            "private plan</think>answer",
            json!([
                {"role":"user","content":"question"},
                {"type":"reasoning","content":"private plan"},
                {"role":"assistant","content":"answer"},
                {"role":"user","content":"and next?"}
            ]),
            true,
        ),
        (
            true,
            false,
            "private plan</think>answer",
            json!([
                {"role":"user","content":"question"},
                {"role":"assistant","content":"answer"},
                {"role":"user","content":"and next?"}
            ]),
            false,
        ),
        (
            true,
            true,
            "private plan</think><\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"lookup\"></\u{ff5c}DSML\u{ff5c} invoke></\u{ff5c}DSML\u{ff5c} calls>",
            json!([
                {"role":"user","content":"question"},
                {"type":"reasoning","content":"private plan"},
                {"type":"function_call","call_id":"c","name":"lookup","arguments":"{ }"},
                {"type":"function_call_output","call_id":"c","output":"found"}
            ]),
            false,
        ),
    ] {
        if tools {
            let mut partition = super::super::protocol_v41::Partition::new(thinking, 1 << 20);
            let mut events = vec![];
            partition.push(output.as_bytes(), &mut events);
            partition.finish(&mut events).unwrap();
            assert!(events.iter().any(|event| matches!(event,
                super::super::partition::PartitionEvent::FunctionCall(call) if call.name == "lookup"
            )));
        }
        let (first, boundaries) = render(
            json!([{"role":"user","content":"question"}]),
            thinking,
            tools,
        );
        let (next, _) = render(turns, thinking, tools);
        let prompt = encode(&first);
        let next = encode(&next);
        let mut consumed = prompt.clone();
        let output = encode(output);
        consumed.extend_from_slice(&output[..output.len() - 1]);
        let matched = decode_loop::extending_prefix(&consumed, &next, consumed.len(), true);
        assert_eq!(
            matched > 0,
            live_expected,
            "thinking={thinking} tools={tools}"
        );
        let cut = encode(&first[..boundaries.generation_header_start.unwrap()]);
        assert!(prompt.starts_with(&cut));
        let mut cache = SnapshotCache::new(1 << 20, Default::default());
        let bytes =
            SnapshotCache::<Vec<u32>>::entry_bytes(cut.len(), (cut.len() * 4) as u64).unwrap();
        assert!(cache.insert_strict(cut.clone(), cut.clone(), bytes));
        let (length, snapshot) = cache
            .best_prefix(&next)
            .expect("safe transcript checkpoint");
        assert_eq!(length, cut.len());
        assert_eq!(&**snapshot, &next[..length]);
        // Decode mutation must not change the independently owned checkpoint.
        consumed.fill(999);
        assert_eq!(&**snapshot, &cut);
    }
}

#[test]
fn renderer_boundary_ignores_marker_lookalikes_in_user_text() {
    let text = format!(
        "literal {}<think> and </think>",
        deepseek_recipe_encoding::v4::ASSISTANT_SP_TOKEN
    );
    let (prompt, boundaries) = render(json!([{"role":"user","content":text}]), true, false);
    let boundary = boundaries.generation_header_start.unwrap();
    assert!(prompt[..boundary].contains(&text));
    assert_eq!(
        &prompt[boundary..],
        format!(
            "{}<think>",
            deepseek_recipe_encoding::v4::ASSISTANT_SP_TOKEN
        )
    );
}

struct Recorded {
    live: Option<reuse::Live<Vec<u32>>>,
    checkpoint: Option<Vec<u32>>,
    checkpoint_logits: Vec<f32>,
    checkpoint_positions: Vec<usize>,
    cache: SnapshotCache<Vec<u32>>,
    restore_checkpoint: bool,
}

impl Default for Recorded {
    fn default() -> Self {
        Self {
            live: None,
            checkpoint: None,
            checkpoint_logits: vec![],
            checkpoint_positions: vec![],
            cache: SnapshotCache::new(1 << 20, Default::default()),
            restore_checkpoint: false,
        }
    }
}

impl reuse::Reuse<Vec<u32>> for Recorded {
    fn take(
        &mut self,
        ids: &[u32],
        _: usize,
    ) -> Result<Option<reuse::Restored<Vec<u32>>>, ServeError> {
        let live = self
            .live
            .take()
            .filter(|live| ids.starts_with(&live.tokens))
            .map(|live| reuse::Restored {
                live,
                source: "live_session",
            });
        Ok(live.or_else(|| {
            if !self.restore_checkpoint {
                return None;
            }
            let (length, checkpoint) = self.cache.best_prefix(ids)?;
            Some(reuse::Restored {
                live: reuse::Live {
                    state: (*checkpoint).clone(),
                    tokens: ids[..length].to_vec(),
                    logits: vec![],
                },
                source: "snapshot_ram",
            })
        }))
    }
    fn checkpoint(&mut self, state: &Vec<u32>, ids: &[u32], logits: &[f32], reserve: u64) {
        assert!(reserve > 0);
        assert_eq!(state, ids);
        self.checkpoint = Some(state.clone());
        self.checkpoint_logits = logits.to_vec();
        self.checkpoint_positions.push(ids.len());
        let bytes =
            SnapshotCache::<Vec<u32>>::entry_bytes(ids.len(), (state.len() * 4) as u64).unwrap();
        self.cache.insert_strict(ids.to_vec(), state.clone(), bytes);
    }
    fn commit(&mut self, live: reuse::Live<Vec<u32>>) {
        self.live = Some(live);
    }
    fn capacity(&self, _: &Vec<u32>, required: usize) -> usize {
        required
    }
}

#[test]
fn execution_retains_only_consumed_tokens_and_final_logits() {
    struct Sink;
    impl GenerationSink for Sink {
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
    }
    let tokenizer = tokenizer();
    let eos = tokenizer.eos().unwrap() as u32;
    let answer = tokenizer.encode("a", false).unwrap()[0] as u32;
    let vocab = tokenizer.n_vocab() as usize;
    let mut settings = Settings::new(
        tokenizer,
        128,
        3,
        SamplingConfig::default(),
        eos,
        "fixture".into(),
    )
    .unwrap();
    settings.headroom = || Some(0);
    for (maximum, stop, fail) in [
        (1, false, false),
        (3, false, false),
        (3, true, false),
        (3, false, true),
    ] {
        let request = settings
            .profile()
            .parse(&json!({"model":"fixture","input":"x","max_output_tokens":maximum}))
            .unwrap();
        let mut recorded = Recorded::default();
        let mut batches = vec![];
        let result = settings.execute(
            &request,
            "prompt",
            None,
            &mut Sink,
            |_, _| Ok(()),
            |_| Ok(vec![]),
            &mut recorded,
            |state, tokens, _, with_logits| {
                assert!(with_logits);
                batches.push(tokens.to_vec());
                state.extend_from_slice(tokens);
                let mut logits = vec![0.; vocab];
                logits[if stop { eos } else { answer } as usize] = 100.;
                Ok(logits)
            },
            |state, token, _| {
                state.push(token);
                if fail {
                    return Err("partial failure".into());
                }
                let mut logits = vec![0.; vocab];
                logits[answer as usize] = 100.;
                Ok(logits)
            },
            || Duration::ZERO,
        );
        let checkpoint = recorded.checkpoint.as_ref().unwrap();
        assert_eq!(checkpoint.len(), 6);
        assert_eq!(batches, [checkpoint.clone()]);
        if fail {
            assert!(result.is_err());
            assert!(recorded.live.is_none());
        } else {
            let outcome = result.unwrap();
            let live = recorded.live.as_ref().unwrap();
            assert_eq!(live.tokens, live.state);
            assert_eq!(live.tokens.len(), 6 + if stop { 0 } else { maximum - 1 });
            assert_eq!(live.logits.len(), vocab);
            assert_eq!(outcome.usage.output_tokens, if stop { 1 } else { maximum });
            if maximum == 1 {
                let replay = settings
                    .execute(
                        &request,
                        "prompt",
                        None,
                        &mut Sink,
                        |_, _| Ok(()),
                        |_| panic!("equal prompt must retain state"),
                        &mut recorded,
                        |_, _, _, _| panic!("equal prompt must not prefill"),
                        |_, _, _| panic!("equal prompt must use last logits"),
                        || Duration::ZERO,
                    )
                    .unwrap();
                assert_eq!(replay.stats.unwrap().matched_tokens, 6);
            }
        }
    }
}

#[test]
fn native_byte_pieces_use_shared_decode_without_lossy_utf8_conversion() {
    #[derive(Default)]
    struct Sink(Vec<Vec<u8>>);
    impl GenerationSink for Sink {
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.0.push(bytes.to_vec());
            Ok(())
        }
    }
    let mut settings = Settings::new(
        tokenizer(),
        128,
        4,
        SamplingConfig::default(),
        tokenizer_fixture::EOS,
        "fixture".into(),
    )
    .unwrap();
    settings.headroom = || Some(0);
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":4}))
        .unwrap();
    let mut sink = Sink::default();
    let mut reuse = Recorded::default();
    let mut script = [0x82, 0xac, tokenizer_fixture::EOS].into_iter();
    let outcome = settings
        .execute(
            &request,
            "p",
            None,
            &mut sink,
            |_, _| Ok(()),
            |_| Ok(vec![]),
            &mut reuse,
            |state, tokens, _, with_logits| {
                assert!(with_logits);
                state.extend_from_slice(tokens);
                let mut logits = vec![0.; 257];
                logits[0xe2] = 100.;
                Ok(logits)
            },
            |state, token, _| {
                state.push(token);
                let mut logits = vec![0.; 257];
                logits[script.next().unwrap() as usize] = 100.;
                Ok(logits)
            },
            || Duration::ZERO,
        )
        .unwrap();
    assert_eq!(sink.0, [vec![0xe2], vec![0x82], vec![0xac]]);
    assert_eq!(String::from_utf8(sink.0.concat()).unwrap(), "\u{20ac}");
    assert_eq!(outcome.usage.output_tokens, 4);
    assert_eq!(reuse.live.unwrap().tokens, [b'p' as u32, 0xe2, 0x82, 0xac]);
}

#[test]
fn cancelled_partial_decode_discards_live_state_but_keeps_prompt_checkpoint() {
    struct Sink(std::sync::Arc<AtomicBool>);
    impl GenerationSink for Sink {
        fn cancellation_flag(&self) -> Option<std::sync::Arc<AtomicBool>> {
            Some(self.0.clone())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
    }
    let mut settings = Settings::new(
        tokenizer(),
        128,
        3,
        SamplingConfig::default(),
        tokenizer_fixture::EOS,
        "fixture".into(),
    )
    .unwrap();
    settings.headroom = || Some(0);
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":3}))
        .unwrap();
    let cancel = std::sync::Arc::new(AtomicBool::new(false));
    let mut reuse = Recorded::default();
    let result = settings.execute(
        &request,
        "p",
        None,
        &mut Sink(cancel.clone()),
        |_, _| Ok(()),
        |_| Ok(vec![]),
        &mut reuse,
        |state, tokens, _, with_logits| {
            assert!(with_logits);
            state.extend_from_slice(tokens);
            let mut logits = vec![0.; 257];
            logits[b'a' as usize] = 100.;
            Ok(logits)
        },
        |state, token, _| {
            state.push(token);
            if state.len() > 1 {
                cancel.store(true, Ordering::Release);
                return Err("cancelled partial token".into());
            }
            let mut logits = vec![0.; 257];
            logits[b'a' as usize] = 100.;
            Ok(logits)
        },
        || Duration::ZERO,
    );
    assert!(
        matches!(result, Err(BackendFailure::Aborted(error)) if error.kind() == io::ErrorKind::Interrupted)
    );
    assert!(reuse.live.is_none());
    assert_eq!(reuse.checkpoint.unwrap(), [b'p' as u32]);
}

fn batch_settings() -> Settings {
    let mut settings = Settings::new(
        tokenizer(),
        256,
        1,
        SamplingConfig::default(),
        tokenizer_fixture::EOS,
        "fixture".into(),
    )
    .unwrap();
    settings.headroom = || Some(0);
    settings
}

fn eos_logits() -> Vec<f32> {
    let mut logits = vec![0.; 257];
    logits[tokenizer_fixture::EOS as usize] = 100.;
    logits
}

struct BatchSink<'a> {
    ticks: &'a std::cell::Cell<usize>,
    cancel: std::sync::Arc<AtomicBool>,
}
impl GenerationSink for BatchSink<'_> {
    fn cancellation_flag(&self) -> Option<std::sync::Arc<AtomicBool>> {
        Some(self.cancel.clone())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.ticks.set(self.ticks.get() + 1);
        Ok(())
    }
    fn piece(&mut self, _: &[u8]) -> io::Result<()> {
        Ok(())
    }
}

fn execute_checkpoint_fixture(
    settings: &Settings,
    request: &ServeRequest,
    prompt: &str,
    boundaries: Option<PromptBoundaries>,
    recorded: &mut Recorded,
) -> (GenerationOutcome, Vec<(usize, usize)>) {
    let ids: Vec<_> = settings
        .tokenizer
        .encode(prompt, false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let ticks = std::cell::Cell::new(0);
    let mut calls = vec![];
    let outcome = settings
        .execute(
            request,
            prompt,
            boundaries,
            &mut BatchSink {
                ticks: &ticks,
                cancel: Default::default(),
            },
            |_, _| Ok(()),
            |_| Ok(vec![]),
            recorded,
            |state, tokens, _, with_logits| {
                let start = state.len();
                state.extend_from_slice(tokens);
                assert_eq!(state, &ids[..state.len()]);
                assert_eq!(with_logits, state.len() == ids.len());
                calls.push((start, state.len()));
                Ok(if with_logits { eos_logits() } else { vec![] })
            },
            |_, _, _| panic!("final logits must select EOS without a decode step"),
            || Duration::ZERO,
        )
        .unwrap();
    let live = recorded.live.as_ref().unwrap();
    assert_eq!(live.state, ids);
    assert_eq!(live.tokens, ids);
    assert_eq!(live.logits, eos_logits());
    assert_eq!(outcome.usage.output_tokens, 1);
    (outcome, calls)
}

#[test]
fn execution_sorts_dedups_and_verifies_both_byte_boundaries() {
    let settings = batch_settings();
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":1}))
        .unwrap();
    let prompt = format!("a{}z", deepseek_recipe_encoding::v4::EOS_TOKEN);
    let header = prompt.len() - 1;
    let inside_utf8 = prompt.find('\u{ff5c}').unwrap() + 1;
    for (shared, generation, cuts) in [
        (Some(1), Some(header), vec![1, 2]),
        (Some(header), Some(1), vec![1, 2]),
        (Some(1), Some(1), vec![1]),
        // Byte-prefix "a<" is not a token prefix: the full marker is one token.
        (Some(2), Some(header), vec![2]),
        (Some(1), Some(2), vec![1]),
        (Some(0), Some(prompt.len()), vec![]),
        (Some(inside_utf8), Some(prompt.len() + 1), vec![]),
        (None, None, vec![]),
    ] {
        let mut recorded = Recorded::default();
        let (_, calls) = execute_checkpoint_fixture(
            &settings,
            &request,
            &prompt,
            Some(PromptBoundaries {
                shared_prefix_end: shared,
                generation_header_start: generation,
            }),
            &mut recorded,
        );
        let endpoints: Vec<_> = cuts.iter().copied().chain([3]).collect();
        assert_eq!(
            calls.iter().map(|&(_, end)| end).collect::<Vec<_>>(),
            endpoints
        );
        assert_eq!(
            recorded.checkpoint_positions,
            if cuts.is_empty() {
                vec![3]
            } else {
                cuts.clone()
            }
        );
        assert_eq!(recorded.checkpoint_logits.is_empty(), !cuts.is_empty());
    }
}

#[test]
fn different_user_restores_official_shared_prelude_with_cache_independent_cuts() {
    let mut settings = batch_settings();
    settings.context = 8192;
    let profile = settings.profile();
    let mut body = json!({"model":"fixture", "input":"task A", "max_output_tokens":1,
        "instructions":"shared system", "reasoning":{"effort":"high"},
        "tools":[{"type":"function", "name":"lookup", "parameters":{"type":"object"}}]});
    let a = profile.parse(&body).unwrap();
    body["input"] = json!("different task B");
    let b = profile.parse(&body).unwrap();
    let encode = |text: &str| {
        settings
            .tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect::<Vec<_>>()
    };
    let mut cold_calls = vec![];
    for budget in [0, 1 << 20] {
        let mut recorded = Recorded {
            cache: SnapshotCache::new(budget, Default::default()),
            restore_checkpoint: true,
            ..Default::default()
        };
        for (index, request) in [&a, &b].into_iter().enumerate() {
            let (prompt, boundaries) = profile.render_prepared(request).unwrap();
            let boundaries = boundaries.unwrap();
            let shared = encode(&prompt[..boundaries.shared_prefix_end.unwrap()]).len();
            let header = encode(&prompt[..boundaries.generation_header_start.unwrap()]).len();
            let ids = encode(&prompt);
            let matched = if index == 1 && budget > 0 { shared } else { 0 };
            recorded.checkpoint_positions.clear();
            let (outcome, calls) = execute_checkpoint_fixture(
                &settings,
                request,
                &prompt,
                Some(boundaries),
                &mut recorded,
            );
            assert_eq!(calls[0].0, matched);
            assert_eq!(calls.last().unwrap().1, ids.len());
            assert_eq!(
                calls.iter().map(|&(start, end)| end - start).sum::<usize>(),
                ids.len() - matched
            );
            assert_eq!(
                recorded.checkpoint_positions,
                if matched == 0 {
                    vec![shared, header]
                } else {
                    vec![header]
                }
            );
            assert!(recorded.checkpoint_logits.is_empty());
            let phases = outcome.stats.unwrap().phases.unwrap();
            assert_eq!(phases["reused_tokens"], matched);
            assert_eq!(phases["prefill_tokens"], ids.len() - matched);
            assert_eq!(
                phases["reuse_source"],
                if matched == 0 { "none" } else { "snapshot_ram" }
            );
            if budget == 0 {
                cold_calls.push(calls);
            } else {
                let expected: Vec<_> = cold_calls[index]
                    .iter()
                    .copied()
                    .filter(|&(start, _)| start >= matched)
                    .collect();
                assert_eq!(
                    calls, expected,
                    "cache admission does not select execution cuts"
                );
            }
        }
        if budget > 0 {
            assert_eq!(
                recorded.cache.len(),
                3,
                "shared prelude and both transcripts survive"
            );
            for (pointer, value) in [
                ("/instructions", json!("changed system")),
                ("/tools/0/name", json!("changed_tool")),
                (
                    "/tools/0/parameters",
                    json!({"type":"object", "properties":{"x":{"type":"string"}}}),
                ),
                ("/reasoning/effort", json!("none")),
                ("/reasoning/effort", json!("low")),
                ("/reasoning/effort", json!("max")),
            ] {
                let mut changed = body.clone();
                *changed.pointer_mut(pointer).unwrap() = value;
                let request = profile.parse(&changed).unwrap();
                let prompt = profile.render(&request).unwrap();
                assert!(
                    recorded.cache.best_prefix(&encode(&prompt)).is_none(),
                    "{pointer}"
                );
            }
            // The official recipe intentionally maps high and xhigh to the same bytes.
            let mut equivalent = body.clone();
            equivalent["reasoning"]["effort"] = json!("xhigh");
            assert_eq!(
                profile
                    .render(&profile.parse(&equivalent).unwrap())
                    .unwrap(),
                profile.render(&b).unwrap()
            );
        }
    }
}

#[test]
fn batched_prefill_splits_verified_checkpoint_and_resumes_live_snapshot_and_equal() {
    let settings = batch_settings();
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":1}))
        .unwrap();
    let mut recorded = Recorded {
        restore_checkpoint: true,
        ..Default::default()
    };
    let ticks = std::cell::Cell::new(0);
    let mut sink = BatchSink {
        ticks: &ticks,
        cancel: Default::default(),
    };
    let cold = "x".repeat(90);
    let live = format!("{cold}abc");
    let changed = format!("{}{}", "x".repeat(45), "y".repeat(45));
    for (prompt, expected, matched, source, rows) in [
        (&cold, vec![(0, 45), (45, 90)], 0, "none", 45),
        (&live, vec![(90, 93)], 90, "live_session", 3),
        (&changed, vec![(45, 90)], 45, "snapshot_ram", 45),
        (&changed, vec![], 90, "live_session", 0),
    ] {
        let ids: Vec<_> = prompt.bytes().map(u32::from).collect();
        let mut calls = vec![];
        let mut last_tick = ticks.get();
        let outcome = settings
            .execute(
                &request,
                prompt,
                Some(PromptBoundaries {
                    generation_header_start: Some(45),
                    shared_prefix_end: None,
                }),
                &mut sink,
                |_, _| Ok(()),
                |_| Ok(vec![]),
                &mut recorded,
                |state, tokens, _, with_logits| {
                    assert!(ticks.get() > last_tick, "transport tick between chunks");
                    last_tick = ticks.get();
                    assert!((1..=crate::family_options::V41_PREFILL_ROWS).contains(&tokens.len()));
                    let start = state.len();
                    assert_eq!(tokens, &ids[start..start + tokens.len()]);
                    state.extend_from_slice(tokens);
                    calls.push((start, state.len()));
                    assert_eq!(with_logits, state.len() == ids.len());
                    Ok(if with_logits { eos_logits() } else { vec![] })
                },
                |_, _, _| panic!("EOS must not be fed"),
                || Duration::ZERO,
            )
            .unwrap();
        assert_eq!(calls, expected);
        let phases = outcome.stats.unwrap().phases.unwrap();
        assert_eq!(phases["reused_tokens"], matched);
        assert_eq!(phases["reuse_source"], source);
        assert_eq!(phases["prefill_rows"], rows);
        assert_eq!(phases["prefill_chunks"], expected.len());
        assert_eq!(phases["prefill_tokens"], ids.len() - matched);
        assert_eq!(
            phases["prefill_mode"],
            if expected.is_empty() {
                "none"
            } else {
                crate::family_options::V41_PREFILL_MODE
            }
        );
        assert_eq!(recorded.checkpoint.as_ref().unwrap(), &ids[..45]);
        assert!(
            recorded.checkpoint_logits.is_empty(),
            "checkpoint chunks are headless"
        );
        let live = recorded.live.as_ref().unwrap();
        assert_eq!(live.tokens, ids);
        assert_eq!(live.state, ids);
        assert_eq!(live.logits, eos_logits());
    }
}

#[test]
fn failed_or_cancelled_chunks_never_publish_partial_state_or_overwrite_checkpoint() {
    let settings = batch_settings();
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":1}))
        .unwrap();
    for (cancelled, completed) in [(false, false), (true, false), (true, true)] {
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let ticks = std::cell::Cell::new(0);
        let mut sink = BatchSink {
            ticks: &ticks,
            cancel: cancel.clone(),
        };
        let mut recorded = Recorded::default();
        let mut calls = 0;
        let result = settings.execute(
            &request,
            &"x".repeat(96),
            Some(PromptBoundaries {
                generation_header_start: Some(16),
                shared_prefix_end: None,
            }),
            &mut sink,
            |_, _| Ok(()),
            |_| Ok(vec![]),
            &mut recorded,
            |state, tokens, _, with_logits| {
                assert_eq!(with_logits, calls == 1);
                calls += 1;
                if calls == 2 {
                    state.extend_from_slice(&tokens[..if completed { tokens.len() } else { 3 }]);
                    cancel.store(cancelled, Ordering::Release);
                    if !completed {
                        return Err(ServeError::server_error("failed partial chunk"));
                    }
                } else {
                    state.extend_from_slice(tokens);
                }
                Ok(vec![])
            },
            |_, _, _| panic!("failed prefill must not decode"),
            || Duration::ZERO,
        );
        assert_eq!(calls, 2, "no work after failed/cancelled chunk");
        assert!(recorded.live.is_none());
        assert_eq!(recorded.checkpoint.unwrap(), vec![b'x' as u32; 16]);
        if cancelled {
            assert!(
                matches!(result, Err(BackendFailure::Aborted(e)) if e.kind() == io::ErrorKind::Interrupted)
            );
        } else {
            assert!(matches!(result, Err(BackendFailure::Serve(e)) if e.status == 500));
        }
    }
}

#[test]
fn headless_and_final_prefill_memory_refusals_stay_typed() {
    let settings = batch_settings();
    let request = settings
        .profile()
        .parse(&json!({"model":"fixture", "input":"x", "max_output_tokens":1}))
        .unwrap();
    for fail_with_logits in [false, true] {
        let ticks = std::cell::Cell::new(0);
        let mut sink = BatchSink {
            ticks: &ticks,
            cancel: Default::default(),
        };
        let mut recorded = Recorded::default();
        let mut calls = vec![];
        let mut admissions = 0;
        let result = settings.execute(
            &request,
            &"x".repeat(65),
            Some(PromptBoundaries {
                generation_header_start: Some(16),
                shared_prefix_end: None,
            }),
            &mut sink,
            |_, _| {
                admissions += 1;
                Ok(())
            },
            |_| Ok(vec![]),
            &mut recorded,
            |state, tokens, _, with_logits| {
                calls.push((tokens.len(), with_logits));
                if with_logits == fail_with_logits {
                    let denied = qwen_llm::metal::MemoryAdmissionDenied {
                        reason: qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient,
                        required_bytes: Some(4096),
                        signals: MetalMemorySignals {
                            recommended_max_bytes: 1024,
                            current_allocated_bytes: 0,
                            process_limit_remaining_bytes: Some(8192),
                        },
                        working_set_headroom_bytes: Some(1024),
                    };
                    return Err(state_error(
                        qwen_llm::deepseek_v41::forward::StateError::MemoryAdmissionDenied(denied),
                    ));
                }
                state.extend_from_slice(tokens);
                Ok(vec![])
            },
            |_, _, _| panic!("memory refusal must precede decode"),
            || Duration::ZERO,
        );
        let Err(BackendFailure::Serve(error)) = result else {
            panic!("expected typed memory refusal");
        };
        assert_eq!(error.status, 503);
        assert_eq!(error.code, Some("memory_admission_denied"));
        assert_eq!(
            admissions, 1,
            "existing request state is not repriced per chunk"
        );
        assert!(recorded.live.is_none());
        if fail_with_logits {
            assert_eq!(calls, [(16, false), (49, true)]);
            assert_eq!(recorded.checkpoint.unwrap(), vec![b'x' as u32; 16]);
            assert!(recorded.checkpoint_logits.is_empty());
        } else {
            assert_eq!(calls, [(16, false)]);
            assert!(recorded.checkpoint.is_none());
        }
    }
}
