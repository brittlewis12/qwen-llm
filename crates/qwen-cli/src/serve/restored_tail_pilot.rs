use super::*;
use qwen_llm::metal_forward::SnapshotKvStorageKind;

fn packed_width(
    dense: bool,
    has_drafter: bool,
    prompt: usize,
    restored: usize,
    exact: bool,
) -> Option<usize> {
    restored_packed_tail_width(dense, has_drafter, true, prompt, restored, exact)
}

#[test]
fn bounded_restored_tail_plan_preserves_unqualified_lanes() {
    for prefix in [8, 8192, 32768] {
        for tail in [1, 6, 7, 16, 32, 33, 48, 49] {
            assert_eq!(
                packed_width(true, false, prefix + tail, prefix, false),
                (7..=32).contains(&tail).then_some(tail)
            );
            assert_eq!(
                packed_width(false, false, prefix + tail, prefix, false),
                None
            );
            assert_eq!(packed_width(true, true, prefix + tail, prefix, false), None);
            assert_eq!(packed_width(true, false, prefix + tail, prefix, true), None);
            assert_eq!(packed_width(true, false, tail, 0, false), None);
            assert_eq!(
                restored_packed_tail_width(true, false, false, prefix + tail, prefix, false),
                None
            );
        }
    }
    assert_eq!(packed_width(true, false, 10, 11, false), None);
    // Seven matches include one pending token: six consumed tokens plus a
    // seven-row suffix is eligible, while incorrectly using matches is not.
    assert_eq!(packed_width(true, false, 13, 6, false), Some(7));
    assert_eq!(packed_width(true, false, 13, 7, false), None);
    let mut arch = qwen_llm::model::QWEN3_27B;
    use qwen_llm::tensor::GgmlType;
    assert!(restored_packed_tail_arch(
        &arch,
        QwenTemplate::Qwen38,
        GgmlType::Q8_0
    ));
    assert!(!restored_packed_tail_arch(
        &arch,
        QwenTemplate::Qwen36,
        GgmlType::Q8_0
    ));
    assert!(!restored_packed_tail_arch(
        &arch,
        QwenTemplate::Qwen38,
        GgmlType::Q6_K
    ));
    arch.mtp_n_hidden_layers = 0;
    assert!(restored_packed_tail_arch(
        &arch,
        QwenTemplate::Qwen38,
        GgmlType::Q8_0
    ));
    arch.n_layer = 63;
    assert!(!restored_packed_tail_arch(
        &arch,
        QwenTemplate::Qwen38,
        GgmlType::Q8_0
    ));
    assert!(!restored_packed_tail_arch(
        &qwen_llm::model::QWEN3_0_8B,
        QwenTemplate::Qwen38,
        GgmlType::Q8_0
    ));
}

#[test]
fn optional_tail_faults_preserve_serial_admission_and_allocation() {
    use qwen_llm::metal::MetalMemorySignals;
    use std::cell::Cell;
    let signals = MetalMemorySignals {
        recommended_max_bytes: 1024,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(32),
    };
    let mut prices = Vec::new();
    let (plan, admission) = admit_optional_tail(Some(("packed", 64)), 0, |price| {
        prices.push(price);
        Ok::<_, &str>(evaluate_metal_memory_admission(price, 8, signals, false))
    })
    .unwrap();
    assert!(plan.is_none());
    assert_eq!(prices, [64, 0]);
    assert_eq!(
        admission,
        evaluate_metal_memory_admission(0, 8, signals, false)
    );
    assert!(admission.admitted);
    let (plan, admission) = admit_optional_tail(Some(("packed", 64)), 0, |price| {
        Ok::<_, &str>(evaluate_metal_memory_admission(
            price,
            8,
            MetalMemorySignals {
                process_limit_remaining_bytes: Some(1),
                ..signals
            },
            false,
        ))
    })
    .unwrap();
    assert!(plan.is_none() && !admission.admitted);
    let (plan, admission) = admit_optional_tail(Some(("packed", 16)), 0, |price| {
        Ok::<_, &str>(evaluate_metal_memory_admission(price, 8, signals, false))
    })
    .unwrap();
    assert_eq!(plan, Some("packed"));
    assert!(admission.admitted);
    assert_eq!(
        admit_optional_tail(Some(("packed", 16)), 0, |_| Err("signal error")),
        Err("signal error")
    );

    let attempts = Cell::new(0);
    let baseline = vec![0x35u8; 64];
    let result = allocate_optional_tail(
        Some("packed"),
        |plan| {
            assert_eq!(plan, "packed");
            attempts.set(attempts.get() + 1);
            Err::<Vec<u8>, _>("injected allocation failure")
        },
        || {
            attempts.set(attempts.get() + 1);
            Ok(baseline.clone())
        },
    )
    .unwrap();
    assert_eq!(result, baseline);
    assert_eq!(attempts.get(), 2);
    let retained = allocate_optional_tail(
        None::<()>,
        |_| -> Result<Vec<u8>, &str> { panic!("unselected candidate") },
        || Ok(baseline.clone()),
    )
    .unwrap();
    assert_eq!(retained, baseline);
    assert_eq!(
        allocate_optional_tail(
            Some(()),
            |_| Ok::<_, &str>(7),
            || panic!("successful candidate retried")
        ),
        Ok(7)
    );
    assert_eq!(
        allocate_optional_tail(Some(()), |_| Err::<(), _>("candidate"), || Err("serial")),
        Err("serial")
    );
}

#[test]
fn consumed_tail_policy_is_explicit_and_work_reduction_scoped() {
    use qwen_llm::tensor::GgmlType;
    let arch = qwen_llm::model::QWEN3_27B;
    let scope = |enabled, template, dtype, drafter, greedy, prompt| {
        consumed_tail_scope(enabled, &arch, template, dtype, drafter, greedy, prompt)
    };
    assert!(scope(
        true,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K,
        false,
        true,
        9013
    ));
    assert!(!scope(
        false,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K,
        false,
        true,
        9013
    ));
    assert!(!scope(
        true,
        QwenTemplate::Qwen38,
        GgmlType::Q6_K,
        false,
        true,
        9013
    ));
    assert!(!scope(
        true,
        QwenTemplate::Qwen36,
        GgmlType::Q8_0,
        false,
        true,
        9013
    ));
    assert!(!scope(
        true,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K,
        true,
        true,
        9013
    ));
    assert!(!scope(
        true,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K,
        false,
        false,
        9013
    ));
    for prompt in [8191, 16385] {
        assert!(!scope(
            true,
            QwenTemplate::Qwen36,
            GgmlType::Q6_K,
            false,
            true,
            prompt
        ));
    }
    let mut wrong = arch;
    wrong.n_layer -= 1;
    assert!(!consumed_tail_scope(
        true,
        &wrong,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K,
        false,
        true,
        9013
    ));
    assert_eq!(consumed_tail_width(9013, 8860, 8987), Some(26));
    for tail in [0, 1, 6, 7, 32, 33, 48] {
        assert_eq!(
            consumed_tail_width(9000 + tail, 8800, 9000),
            (7..=32).contains(&tail).then_some(tail)
        );
    }
    for (prompt, baseline, consumed) in [
        (9013, 9014, 8987),
        (9013, 8860, 9014),
        (9013, 8980, 8987),
        (9013, 8924, 8987),
    ] {
        assert_eq!(consumed_tail_width(prompt, baseline, consumed), None);
    }
    assert_eq!(consumed_tail_width(9013, 8923, 8987), Some(26));
}

#[test]
fn consumed_request_faults_drop_candidate_before_larger_baseline_admission() {
    use std::cell::RefCell;
    use std::rc::Rc;
    struct Allocation(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Allocation {
        fn drop(&mut self) {
            self.0.borrow_mut().push("drop_candidate");
        }
    }
    for fault in ["admission_error", "denied", "allocation", "restore", "none"] {
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = try_optional_request(
            50,
            |&price| {
                assert_eq!(price, 50);
                events.borrow_mut().push("admit_small");
                if fault == "admission_error" {
                    Err("admission")
                } else {
                    Ok(fault != "denied")
                }
            },
            |_| {
                events.borrow_mut().push("allocate");
                let allocation = Allocation(Rc::clone(&events));
                if fault == "allocation" {
                    Err("allocation")
                } else {
                    Ok(allocation)
                }
            },
            |allocation| {
                events.borrow_mut().push("restore");
                if fault == "restore" {
                    Err("restore")
                } else {
                    Ok(allocation)
                }
            },
        );
        if fault == "none" {
            assert!(matches!(&result, Ok(Some(_))));
            assert_eq!(*events.borrow(), ["admit_small", "allocate", "restore"]);
            drop(result);
        } else {
            assert!(!matches!(&result, Ok(Some(_))));
            let before = events.borrow().clone();
            if ["allocation", "restore"].contains(&fault) {
                assert_eq!(before.last(), Some(&"drop_candidate"));
            }
            events.borrow_mut().push("admit_large_baseline");
            // The old plan can be denied even when the 50-byte candidate fit.
            let signals = qwen_llm::metal::MetalMemorySignals {
                current_allocated_bytes: 0,
                recommended_max_bytes: 1024,
                process_limit_remaining_bytes: Some(1024),
            };
            assert!(!evaluate_metal_memory_admission(1300, 8, signals, false).admitted);
            assert_eq!(events.borrow().last(), Some(&"admit_large_baseline"));
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|v| v.is_finite()));
    let dot: f64 = a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    let (aa, bb) = (norm(a), norm(b));
    if aa == 0.0 && bb == 0.0 {
        1.0
    } else {
        dot / (aa * bb).sqrt()
    }
}

#[test]
#[ignore = "CPU-only; requires QWEN_REPLAY_MODEL and QWEN_REPLAY_RECORDING"]
fn recorded_render_boundary_diagnostic() {
    let model = std::env::var("QWEN_REPLAY_MODEL").unwrap();
    let recording = std::path::PathBuf::from(std::env::var("QWEN_REPLAY_RECORDING").unwrap());
    let index: usize = std::env::var("QWEN_REPLAY_REQUEST_INDEX")
        .expect("set QWEN_REPLAY_REQUEST_INDEX to the prior request's zero-based index")
        .parse()
        .unwrap();
    let gguf = GgufFile::open(&model).unwrap();
    eprintln!(
        "render-boundary output_tensor={:?}",
        gguf.tensors
            .iter()
            .find(|tensor| tensor.name == "output.weight")
    );
    let family = qwen_llm::model_family::ModelFamily::detect(&gguf).unwrap();
    let template = crate::prompt_template::serve_qwen_template(family, &gguf).unwrap();
    let tokenizer = Tokenizer::open(&model).unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(recording.join("rows.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(
        index < rows.len().saturating_sub(1),
        "requires two adjacent recorded requests"
    );
    let render = |index| {
        let value = serde_json::from_str(
            &std::fs::read_to_string(recording.join(format!("request-{index}.json"))).unwrap(),
        )
        .unwrap();
        let request = crate::open_responses::items::parse_request(&value).unwrap();
        let request = crate::open_responses::bind_qwen_request(
            &request,
            template,
            crate::supports_qwen_no_thinking_prompt(family, &gguf),
        )
        .unwrap();
        crate::open_responses::render::render_qwen_serve_prompt(&request)
    };
    let prior = render(index);
    let next = render(index + 1);
    let mut prior_ids = tokenizer.encode(&prior, false).unwrap();
    let next_ids = tokenizer.encode(&next, false).unwrap();
    assert_eq!(
        prior_ids.len() as u64,
        rows[index]["response"]["usage"]["input_tokens"]
            .as_u64()
            .unwrap()
    );
    assert_eq!(
        next_ids.len() as u64,
        rows[index + 1]["response"]["usage"]["input_tokens"]
            .as_u64()
            .unwrap()
    );
    let text = rows[index]["text"].as_str().unwrap();
    let canonical_output = tokenizer.encode(text, false).unwrap();
    let output_count = canonical_output.len();
    prior_ids.extend(canonical_output);
    let lcp = prior_ids
        .iter()
        .zip(&next_ids)
        .take_while(|(a, b)| a == b)
        .count();
    let trim_chars = text.chars().count() - text.trim_end().chars().count();
    eprintln!(
        "render-boundary template={template:?} recorded_output_tokens={} canonical_output_tokens={output_count} reconstructed_key={} next_prompt={} lcp={lcp} trailing_whitespace={trim_chars}",
        rows[index]["response"]["usage"]["output_tokens"],
        prior_ids.len(),
        next_ids.len()
    );
    if let (Some(&old), Some(&new)) = (prior_ids.get(lcp), next_ids.get(lcp)) {
        eprintln!(
            "render-boundary first_difference old={old} {:?} new={new} {:?}",
            tokenizer.decode(&[old]),
            tokenizer.decode(&[new])
        );
    }
    eprintln!(
        "render-boundary verbatim_prefix={} trimmed_prefix={} canonical_consumed_prefix_matches={}",
        next.starts_with(&format!("{prior}{text}")),
        next.starts_with(&format!("{prior}{}", text.trim_end())),
        !prior_ids.is_empty() && lcp >= prior_ids.len() - 1
    );
    eprintln!(
        "render-boundary diagnostic only: canonical output IDs are not a witness of actual emitted IDs or stored checkpoint state"
    );
}

#[test]
#[ignore = "serial Metal replay; requires QWEN_REPLAY_MODEL, QWEN_REPLAY_RECORDING, QWEN_REPLAY_REQUEST_INDEX"]
fn recorded_completed_checkpoint_witness() {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let model = std::env::var("QWEN_REPLAY_MODEL").unwrap();
    let recording = std::path::PathBuf::from(std::env::var("QWEN_REPLAY_RECORDING").unwrap());
    let index: usize = std::env::var("QWEN_REPLAY_REQUEST_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(recording.join("rows.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(index < rows.len().saturating_sub(1));
    let request = |i| {
        let value = serde_json::from_str(
            &std::fs::read_to_string(recording.join(format!("request-{i}.json"))).unwrap(),
        )
        .unwrap();
        crate::open_responses::items::parse_request(&value).unwrap()
    };
    let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
    let loaded = runtime.load_model(&model).unwrap();
    assert_eq!(loaded.prefix_cache_stats().entries, 0);
    let family = qwen_llm::model_family::ModelFamily::detect(loaded.gguf()).unwrap();
    let template = crate::prompt_template::serve_qwen_template(family, loaded.gguf()).unwrap();
    let no_thinking = crate::supports_qwen_no_thinking_prompt(family, loaded.gguf());
    let mut backend = EngineBackend::new(
        loaded,
        "boundary-witness".into(),
        128,
        Some(16384),
        16384,
        None,
        template,
        no_thinking,
    )
    .unwrap();
    let mut completed_ids = Vec::new();
    for (i, row) in rows.iter().enumerate().take(index + 1) {
        let request = request(i);
        assert_eq!(request_sampler(&request).unwrap().config().temperature, 0.0);
        let prompt = backend.render_prompt(&request).unwrap();
        let prompt_ids = backend.tokenizer.encode(&prompt, false).unwrap();
        if i == index {
            completed_ids = prompt_ids.clone();
            completed_ids.extend(
                backend
                    .tokenizer
                    .encode(row["text"].as_str().unwrap(), false)
                    .unwrap(),
            );
            assert!(
                backend
                    .loaded
                    .lookup_cached_prefix(&completed_ids)
                    .is_none_or(|hit| hit.restored_prefix_len() < completed_ids.len() - 1)
            );
        }
        let outcome = backend
            .generate(&request, &prompt, &mut Sink)
            .unwrap_or_else(|_| panic!("replay request {i} failed"));
        assert_eq!(
            outcome.usage.input_tokens as u64,
            row["response"]["usage"]["input_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            outcome.usage.output_tokens as u64,
            row["response"]["usage"]["output_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            outcome.usage.cached_tokens as u64,
            row["response"]["usage"]["input_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap()
        );
    }
    let lookup = backend
        .loaded
        .lookup_cached_prefix(&completed_ids)
        .expect("new completed checkpoint");
    assert_eq!(lookup.restored_prefix_len(), completed_ids.len() - 1);
    assert!(!lookup.is_exact_with_final_logits());
    let mut sequence = backend
        .loaded
        .create_sequence(SequenceConfig::new(16384))
        .unwrap();
    let report = backend
        .loaded
        .restore_prepared_cached_prefix(lookup, &mut sequence, &completed_ids)
        .unwrap();
    assert!(report.exact);
    assert_eq!(report.matched_prefix_len, completed_ids.len());
    assert_eq!(report.restored_prefix_len, completed_ids.len() - 1);
    assert_eq!(sequence.position(), completed_ids.len() - 1);
    assert!(report.exact_final_logits.is_none());
    let next_prompt = backend.render_prompt(&request(index + 1)).unwrap();
    let next_ids = backend.tokenizer.encode(&next_prompt, false).unwrap();
    let lcp = completed_ids
        .iter()
        .zip(&next_ids)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        lcp >= report.restored_prefix_len,
        "next prompt changes consumed state"
    );
    let next_restored = backend
        .loaded
        .lookup_cached_prefix(&next_ids)
        .map_or(0, |hit| hit.restored_prefix_len());
    eprintln!(
        "checkpoint-witness template={template:?} matched={} restored={} pending={} next_prompt={} lcp={lcp} next_restored={next_restored} required_forwards={} consumed_only_forwards={}",
        report.matched_prefix_len,
        report.restored_prefix_len,
        completed_ids.last().unwrap(),
        next_ids.len(),
        next_ids.len() - next_restored,
        next_ids.len() - report.restored_prefix_len
    );
    drop(sequence);
    if std::env::var_os("QWEN_REPLAY_COMPARE_TAILS").is_some() {
        compare_consumed_boundary_tails(&backend.loaded, &completed_ids, &next_ids);
    }
}

fn compare_consumed_boundary_tails(loaded: &LoadedModel, completed: &[i32], next: &[i32]) {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            panic!("prefill emitted")
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let consumed = completed.len() - 1;
    assert!(next.starts_with(&completed[..consumed]));
    assert!((7..=32).contains(&(next.len() - consumed)));
    let forward = loaded.forward();
    let stops = loaded.gguf().stop_token_ids().unwrap();
    let mut reference_logits: Option<Vec<f32>> = None;
    let mut reference_tokens = None;
    let mut reference_state: Option<qwen_llm::metal_forward::SessionSnapshot> = None;
    let mut unchanged_prefix = 0;
    for mode in ["old-packed", "consumed-serial", "consumed-packed"] {
        let key = if mode == "old-packed" {
            next
        } else {
            completed
        };
        let lookup = (mode == "old-packed").then(|| loaded.lookup_cached_prefix(key).unwrap());
        let consumed_lookup =
            (mode != "old-packed").then(|| loaded.lookup_cached_consumed_extension(next).unwrap());
        let before = loaded.context().current_allocated_size();
        let start = Instant::now();
        let (chunk, mut scratch, mut sequence) = if mode == "consumed-packed" {
            let width = next.len() - consumed;
            let plan = qwen_llm::metal_dflash::plan_single_chunk_prefill_scratch(
                loaded.metal_model(),
                width as u32,
                next.len(),
                PrefillScratchConfig::default(),
            )
            .unwrap();
            let price = plan
                .priced_upper_bound(|bytes| {
                    Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size)
                })
                .unwrap();
            assert!(price <= 128 * 1024 * 1024);
            let scratch = MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
                loaded.context(),
                loaded.metal_model(),
                plan,
            )
            .unwrap();
            let scratch_bytes = loaded.context().current_allocated_size() - before;
            assert!(scratch_bytes <= 128 * 1024 * 1024);
            eprintln!(
                "consumed-tail mode={mode} scratch_bytes={scratch_bytes} priced_bytes={price}"
            );
            (
                width,
                Some(scratch),
                loaded
                    .create_sequence(SequenceConfig::new(next.len() + 128))
                    .unwrap(),
            )
        } else {
            allocate_serve_request_state(loaded, next.len(), next.len() + 128, mode == "old-packed")
                .unwrap()
        };
        let allocation_ms = start.elapsed().as_secs_f64() * 1e3;
        let request_bytes = loaded.context().current_allocated_size() - before;
        let start = Instant::now();
        let restored = if let Some(lookup) = consumed_lookup {
            assert_eq!(lookup.restored_prefix_len(), consumed);
            let mut wrong_prefix = next.to_vec();
            wrong_prefix[0] ^= 1;
            for rejected in [&wrong_prefix[..], &next[..consumed], completed] {
                let bad_lookup = loaded.lookup_cached_consumed_extension(next).unwrap();
                assert!(
                    loaded
                        .restore_prepared_consumed_extension(bad_lookup, &mut sequence, rejected)
                        .is_err()
                );
                assert_eq!(sequence.position(), 0);
            }
            let report = loaded
                .restore_prepared_consumed_extension(lookup, &mut sequence, next)
                .unwrap();
            assert_eq!(report.matched_prefix_len, consumed);
            assert_eq!(report.restored_prefix_len, consumed);
            assert!(!report.exact);
            assert!(report.exact_final_logits.is_none());
            assert!(report.capture_tail.is_none());
            report
        } else {
            loaded
                .restore_prepared_cached_prefix(lookup.unwrap(), &mut sequence, key)
                .unwrap()
        };
        assert!(next.starts_with(&key[..restored.restored_prefix_len]));
        let restore_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let mut logits = prefill_remaining(
            loaded,
            &forward,
            None,
            false,
            next,
            chunk,
            &mut sequence,
            &mut scratch,
            &mut None,
            &mut Sink,
        )
        .unwrap_or_else(|_| panic!("{mode} prefill failed"))
        .unwrap();
        let prefill_ms = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(sequence.position(), next.len());
        if let Some(reference) = &reference_logits {
            let cos = cosine(reference, &logits);
            eprintln!("consumed-tail mode={mode} logits_cos={cos:.10}");
            assert!(cos >= 0.999);
        } else {
            reference_logits = Some(logits.clone());
        }
        let snapshot = sequence
            .metal_session()
            .snapshot(
                loaded.snapshot_identity(&sequence).unwrap(),
                next.to_vec(),
                None,
            )
            .unwrap();
        if let Some(reference) = &reference_state {
            let cos = consumed_tail_state_cosine(reference, &snapshot, unchanged_prefix);
            eprintln!("consumed-tail mode={mode} min_state_cos={cos:.10}");
            assert!(cos >= 0.999);
        } else {
            unchanged_prefix = restored.restored_prefix_len;
            reference_state = Some(snapshot);
        }
        let request = ServeRequest {
            temperature: Some(0.0),
            ..ServeRequest::default()
        };
        let mut sampler = request_sampler(&request).unwrap();
        let start = Instant::now();
        let mut tokens = Vec::new();
        for step in 0..128 {
            let token = sampler.sample(&logits).unwrap().token;
            tokens.push(token);
            if stops.contains(&token) || step == 127 {
                break;
            }
            logits = forward
                .single_token(token, sequence.position() as u32, unsafe {
                    sequence.metal_session_mut()
                })
                .unwrap();
            sequence.advance_by(1).unwrap();
        }
        let decode_ms = start.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "consumed-tail mode={mode} restored={} rows={} request_bytes={request_bytes} allocation_ms={allocation_ms:.3} restore_ms={restore_ms:.3} prefill_ms={prefill_ms:.3} decode_ms={decode_ms:.3} tokens={} sha256={}",
            restored.restored_prefix_len,
            next.len() - restored.restored_prefix_len,
            tokens.len(),
            token_ids_sha256_i32le(&tokens)
        );
        if let Some(reference) = &reference_tokens {
            assert_eq!(reference, &tokens);
        } else {
            reference_tokens = Some(tokens);
        }
    }
}

fn consumed_tail_state_cosine(
    a: &qwen_llm::metal_forward::SessionSnapshot,
    b: &qwen_llm::metal_forward::SessionSnapshot,
    unchanged_prefix: usize,
) -> f64 {
    assert_eq!(a.identity, b.identity);
    assert_eq!(a.kv_n_pos, b.kv_n_pos);
    assert_eq!(a.identity.kv_storage_kind, SnapshotKvStorageKind::F16);
    let layer_bytes = a.prefix_tokens.len() * a.identity.kv_bytes_per_token as usize;
    let old_bytes = unchanged_prefix * a.identity.kv_bytes_per_token as usize;
    let f16_values = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>()
    };
    let mut min_cos = 1.0f64;
    for (x, y) in [
        (&a.kv_k_arena, &b.kv_k_arena),
        (&a.kv_v_arena, &b.kv_v_arena),
    ] {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len() % layer_bytes, 0);
        for (x, y) in x.chunks_exact(layer_bytes).zip(y.chunks_exact(layer_bytes)) {
            assert_eq!(&x[..old_bytes], &y[..old_bytes]);
            min_cos = min_cos.min(cosine(
                &f16_values(&x[old_bytes..]),
                &f16_values(&y[old_bytes..]),
            ));
        }
    }
    for (x, y, elements) in [
        (
            &a.gdn_state_arena,
            &b.gdn_state_arena,
            a.identity.gdn_state_elements_per_layer,
        ),
        (
            &a.gdn_conv_arena,
            &b.gdn_conv_arena,
            a.identity.gdn_conv_elements_per_layer,
        ),
    ] {
        assert_eq!(x.len(), y.len());
        let layer_bytes = elements as usize * 4;
        assert_eq!(x.len() % layer_bytes, 0);
        for (x, y) in x.chunks_exact(layer_bytes).zip(y.chunks_exact(layer_bytes)) {
            min_cos = min_cos.min(cosine(&f32_values(x), &f32_values(y)));
        }
    }
    min_cos
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires QWEN_TAIL_PILOT_MODEL and QWEN_TAIL_PILOT_PROMPT; serial Metal pilot"]
fn restored_suffix32_packed_matches_serial_greedy() {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            panic!("prefill emitted")
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let model = std::env::var("QWEN_TAIL_PILOT_MODEL").unwrap();
    let prompt = std::fs::read_to_string(std::env::var("QWEN_TAIL_PILOT_PROMPT").unwrap()).unwrap();
    let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
    let loaded = runtime.load_model(&model).unwrap();
    assert_eq!(loaded.arch().kind, qwen_llm::model::ArchKind::Dense);
    let tokens = loaded.tokenizer().unwrap().encode(&prompt, false).unwrap();
    let prefix = tokens.len().checked_sub(32).unwrap();
    assert!(prefix >= 8192);
    let width = packed_width(true, false, tokens.len(), prefix, false).unwrap();
    let forward = loaded.forward();
    let (_, mut prefix_scratch, mut builder) =
        allocate_serve_request_state(&loaded, prefix, tokens.len() + 64, true).unwrap();
    crate::prefill_span(
        &forward,
        &mut builder,
        prefix_scratch.as_mut().unwrap(),
        &tokens[..prefix],
        0,
    )
    .unwrap();
    let checkpoint = loaded
        .prepare_checkpoint_boundary(
            &builder,
            tokens[..prefix].to_vec(),
            Some(tokens[prefix]),
            None,
            None,
            0,
        )
        .unwrap();
    loaded
        .cache_prepared_checkpoint_strict(&checkpoint)
        .unwrap()
        .expect("cache insertion");
    drop(builder);
    drop(prefix_scratch);

    let mut states = Vec::new();
    let mut logits = Vec::new();
    let mut times = Vec::new();
    let mut reusable_scratch = None;
    for mode in ["serial", "full-vt", "single-vt", "single-vt-reuse"] {
        let packed = mode != "serial";
        let single = mode.starts_with("single-vt");
        let mut sequence = loaded
            .create_sequence(SequenceConfig::new(tokens.len() + 64))
            .unwrap();
        let lookup = loaded.lookup_cached_prefix(&tokens).unwrap();
        assert_eq!(lookup.restored_prefix_len(), prefix);
        let report = loaded
            .restore_prepared_cached_prefix(lookup, &mut sequence, &tokens)
            .unwrap();
        assert_eq!(report.matched_prefix_len, prefix + 1);
        assert_eq!(sequence.position(), prefix);
        let before = loaded.context().current_allocated_size();
        let start = Instant::now();
        let mut scratch = if mode == "single-vt-reuse" {
            reusable_scratch.take()
        } else if packed {
            let planner = if single {
                qwen_llm::metal_dflash::plan_single_chunk_prefill_scratch
            } else {
                plan_prefill_scratch_with_matrix_max_pos_configured
            };
            let plan = planner(
                loaded.metal_model(),
                width as u32,
                tokens.len(),
                PrefillScratchConfig::default(),
            )
            .unwrap();
            let priced = plan
                .priced_upper_bound(|bytes| {
                    Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size)
                })
                .unwrap();
            eprintln!("tail-pilot mode={mode} priced_upper_bytes={priced}");
            if single {
                assert!(priced <= 128 * 1024 * 1024);
            }
            Some(
                MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
                    loaded.context(),
                    loaded.metal_model(),
                    plan,
                )
                .unwrap(),
            )
        } else {
            None
        };
        let bytes = loaded.context().current_allocated_size() - before;
        if let Some(scratch) = &scratch {
            let plan = scratch.prefill_scratch_plan();
            assert_eq!(plan.block_size(), width as u32);
            assert_eq!(plan.matrix_query_rows(), width as u32);
            assert_eq!(plan.matrix_max_pos(), tokens.len() as u64);
        }
        let allocation_ms = start.elapsed().as_secs_f64() * 1e3;
        if single {
            let scratch = scratch.as_mut().unwrap();
            let before = sequence
                .metal_session()
                .snapshot(
                    loaded.snapshot_identity(&sequence).unwrap(),
                    tokens[..prefix].to_vec(),
                    None,
                )
                .unwrap();
            for bad in [&tokens[prefix..prefix], &tokens[prefix - 1..]] {
                let error = qwen_llm::metal_dflash::prefill_tokens_with_multi_hidden(
                    &forward,
                    bad,
                    prefix as u32,
                    unsafe { sequence.metal_session_mut() },
                    scratch,
                    &[],
                    None,
                )
                .unwrap_err();
                assert!(error.to_string().contains("single-chunk"));
            }
            let after = sequence
                .metal_session()
                .snapshot(
                    loaded.snapshot_identity(&sequence).unwrap(),
                    tokens[..prefix].to_vec(),
                    None,
                )
                .unwrap();
            assert_eq!(before.kv_n_pos, after.kv_n_pos);
            assert_eq!(before.kv_k_arena, after.kv_k_arena);
            assert_eq!(before.kv_v_arena, after.kv_v_arena);
            assert_eq!(before.gdn_state_arena, after.gdn_state_arena);
            assert_eq!(before.gdn_conv_arena, after.gdn_conv_arena);
            if mode == "single-vt-reuse" {
                let vt = &scratch.attn_matrix_vt_pack;
                unsafe {
                    std::ptr::write_bytes(
                        (vt.buffer.contents().as_ptr() as *mut u8).add(vt.offset as usize),
                        0x7e,
                        vt.n_bytes() as usize,
                    );
                }
            }
        }
        let start = Instant::now();
        let out = if packed {
            crate::prefill_span(
                &forward,
                &mut sequence,
                scratch.as_mut().unwrap(),
                &tokens[prefix..],
                prefix,
            )
            .unwrap()
            .0
        } else {
            prefill_remaining(
                &loaded,
                &forward,
                None,
                false,
                &tokens,
                width,
                &mut sequence,
                &mut scratch,
                &mut None,
                &mut Sink,
            )
            .unwrap_or_else(|_| panic!("serial prefill"))
            .unwrap()
        };
        let ms = start.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "tail-pilot mode={mode} prefix={prefix} suffix={width} scratch_bytes={bytes} allocation_ms={allocation_ms:.3} prefill_ms={ms:.3}"
        );
        assert!(
            !single || bytes <= 128 * 1024 * 1024,
            "suffix scratch exceeds 128 MiB screen"
        );
        assert_eq!(sequence.position(), tokens.len());
        times.push(ms + allocation_ms);
        logits.push(out);
        states.push(sequence);
        if mode == "single-vt" {
            reusable_scratch = scratch;
        }
    }
    for other in [2, 3] {
        assert!(
            logits[1]
                .iter()
                .zip(&logits[other])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "VT sharing changed packed logits"
        );
    }
    let logits_cos = cosine(&logits[0], &logits[2]);
    eprintln!(
        "tail-pilot logits_cos={logits_cos:.10} phase_speedup={:.3}",
        times[0] / times[2]
    );
    assert!(logits_cos >= 0.999);
    let snapshots: Vec<_> = states
        .iter()
        .map(|s| {
            s.metal_session()
                .snapshot(loaded.snapshot_identity(s).unwrap(), tokens.clone(), None)
                .unwrap()
        })
        .collect();
    for other in [2, 3] {
        let (a, b) = (&snapshots[1], &snapshots[other]);
        assert_eq!(a.kv_n_pos, b.kv_n_pos);
        assert_eq!(a.kv_k_arena, b.kv_k_arena);
        assert_eq!(a.kv_v_arena, b.kv_v_arena);
        assert_eq!(a.gdn_state_arena, b.gdn_state_arena);
        assert_eq!(a.gdn_conv_arena, b.gdn_conv_arena);
    }
    eprintln!(
        "tail-pilot full-VT/single-VT/poisoned-reuse logits and persistent state bitwise equal"
    );
    let (a, b) = (&snapshots[0], &snapshots[2]);
    assert_eq!(a.kv_n_pos, b.kv_n_pos);
    assert_eq!(a.identity, b.identity);
    assert_eq!(a.identity.kv_storage_kind, SnapshotKvStorageKind::F16);
    let layer_bytes = tokens.len() * a.identity.kv_bytes_per_token as usize;
    let restored_bytes = prefix * a.identity.kv_bytes_per_token as usize;
    let f16_values = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>()
    };
    let mut min_cos = 1.0f64;
    for (x, y) in [
        (&a.kv_k_arena, &b.kv_k_arena),
        (&a.kv_v_arena, &b.kv_v_arena),
    ] {
        for (x, y) in x.chunks_exact(layer_bytes).zip(y.chunks_exact(layer_bytes)) {
            assert_eq!(&x[..restored_bytes], &y[..restored_bytes]);
            min_cos = min_cos.min(cosine(
                &f16_values(&x[restored_bytes..]),
                &f16_values(&y[restored_bytes..]),
            ));
        }
    }
    for (x, y, elements) in [
        (
            &a.gdn_state_arena,
            &b.gdn_state_arena,
            a.identity.gdn_state_elements_per_layer,
        ),
        (
            &a.gdn_conv_arena,
            &b.gdn_conv_arena,
            a.identity.gdn_conv_elements_per_layer,
        ),
    ] {
        for (x, y) in x
            .chunks_exact(elements as usize * 4)
            .zip(y.chunks_exact(elements as usize * 4))
        {
            min_cos = min_cos.min(cosine(&f32_values(x), &f32_values(y)));
        }
    }
    eprintln!("tail-pilot min_state_cos={min_cos:.10}");
    assert!(min_cos >= 0.999);
    drop(snapshots);
    let stops = loaded.gguf().stop_token_ids().unwrap();
    let request = ServeRequest {
        temperature: Some(0.0),
        ..ServeRequest::default()
    };
    let mut sampler = request_sampler(&request).unwrap();
    let mut generated = Vec::new();
    for step in 0..64 {
        let next: Vec<_> = logits
            .iter()
            .map(|logits| sampler.sample(logits).unwrap().token)
            .collect();
        assert_eq!(
            next[0], next[1],
            "greedy continuation differs at step {step}"
        );
        assert!(next.iter().all(|&token| token == next[0]));
        generated.push(next[0]);
        if stops.contains(&next[0]) || step == 63 {
            break;
        }
        for (s, logits) in states.iter_mut().zip(&mut logits) {
            *logits = forward
                .single_token(next[0], s.position() as u32, unsafe {
                    s.metal_session_mut()
                })
                .unwrap();
            s.advance_by(1).unwrap();
        }
    }
    assert_eq!(states[0].position(), states[1].position());
    eprintln!(
        "tail-pilot greedy_tokens={} sha256={}",
        generated.len(),
        token_ids_sha256_i32le(&generated)
    );
}
