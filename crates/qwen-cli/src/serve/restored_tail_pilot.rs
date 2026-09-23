use super::*;
use qwen_llm::metal_forward::SnapshotKvStorageKind;

fn packed_width(
    dense: bool,
    has_drafter: bool,
    prompt: usize,
    restored: usize,
    exact: bool,
) -> Option<usize> {
    restored_packed_tail_width(dense, has_drafter, prompt, restored, exact)
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
        }
    }
    assert_eq!(packed_width(true, false, 10, 11, false), None);
    // Seven matches include one pending token: six consumed tokens plus a
    // seven-row suffix is eligible, while incorrectly using matches is not.
    assert_eq!(packed_width(true, false, 13, 6, false), Some(7));
    assert_eq!(packed_width(true, false, 13, 7, false), None);
    // Eligibility is the 27B dense geometry only. Template (3.5/3.6/3.8),
    // weight or LM-head dtype, and greedy vs sampled no longer exclude a
    // request: they don't change what the packed kernels compute.
    let mut arch = qwen_llm::model::QWEN3_27B;
    assert!(bounded_packed_dense_arch(&arch));
    arch.mtp_n_hidden_layers = 0;
    assert!(bounded_packed_dense_arch(&arch));
    arch.n_layer = 63;
    assert!(!bounded_packed_dense_arch(&arch));
    assert!(!bounded_packed_dense_arch(&qwen_llm::model::QWEN3_0_8B));
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
fn fresh_packed_policy_preserves_cached_sampled_and_unqualified_requests() {
    use qwen_llm::tensor::GgmlType;
    let arch = qwen_llm::model::QWEN3_27B;
    assert!(!fresh_packed_arch(
        &arch,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K
    ));
    assert!(fresh_packed_arch(
        &arch,
        QwenTemplate::Qwen38,
        GgmlType::Q8_0
    ));
    for (template, dtype) in [
        (QwenTemplate::Qwen36, GgmlType::Q8_0),
        (QwenTemplate::Qwen38, GgmlType::Q6_K),
        (QwenTemplate::Qwen35, GgmlType::Q6_K),
    ] {
        assert!(!fresh_packed_arch(&arch, template, dtype));
    }
    let mut wrong = arch;
    wrong.n_layer -= 1;
    assert!(!fresh_packed_arch(
        &wrong,
        QwenTemplate::Qwen36,
        GgmlType::Q6_K
    ));
    for prompt in [0, 1, 18, 19, 20, 31, 32, 47, 48, 49, 8192] {
        assert_eq!(
            fresh_packed_width(true, true, false, true, false, prompt),
            (19..=48).contains(&prompt).then_some(prompt)
        );
        assert_eq!(
            fresh_packed_width(false, true, false, true, false, prompt),
            None
        );
        assert_eq!(
            fresh_packed_width(true, false, false, true, false, prompt),
            None
        );
        assert_eq!(
            fresh_packed_width(true, true, true, true, false, prompt),
            None
        );
        assert_eq!(
            fresh_packed_width(true, true, false, false, false, prompt),
            None
        );
        assert_eq!(
            fresh_packed_width(true, true, false, true, true, prompt),
            None
        );
    }
}

#[test]
fn fresh_packed_opt_in_and_rollback_are_fail_closed() {
    use std::ffi::OsStr;
    assert!(!fresh_packed_enabled(None));
    assert!(fresh_packed_enabled(Some(OsStr::new("1"))));
    for value in ["0", "", "true", "yes", "1 ", "01", "invalid"] {
        assert!(!fresh_packed_enabled(Some(OsStr::new(value))));
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(!fresh_packed_enabled(Some(OsStr::from_bytes(&[0xff]))));
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
        let lookup = loaded.lookup_cached_prefix(key).unwrap();
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
        let restored = loaded
            .restore_prepared_cached_prefix(lookup, &mut sequence, key)
            .unwrap();
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
#[ignore = "serial Metal pilot; requires QWEN_FRESH_PILOT_MODEL"]
fn fresh_short_packed_matches_serial() {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            panic!("prefill emitted")
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
    let loaded = runtime
        .load_model(std::env::var("QWEN_FRESH_PILOT_MODEL").unwrap())
        .unwrap();
    let tokenizer = loaded.tokenizer().unwrap();
    let family = qwen_llm::model_family::ModelFamily::detect(loaded.gguf()).unwrap();
    let template = crate::prompt_template::serve_qwen_template(family, loaded.gguf()).unwrap();
    let forward = loaded.forward();
    let stops = loaded.gguf().stop_token_ids().unwrap();
    for (width, text) in [
        (19, "Reply with exactly the word violet."),
        (
            32,
            "Write Python that merges two sorted lists without duplicates.",
        ),
        (
            48,
            "Explain how a binary search tree stores and finds values.",
        ),
    ] {
        let tokens = (0..64).find_map(|padding| {
                let value = serde_json::json!({"model": "fresh-pilot", "input": format!("{}{text}", "Hello ".repeat(padding)),
                "max_output_tokens": 64, "temperature": 0, "x_qwen": {"no_thinking": true}});
            let request = crate::open_responses::items::parse_request(&value).unwrap();
            let request = crate::open_responses::bind_qwen_request(&request, template,
                crate::supports_qwen_no_thinking_prompt(family, loaded.gguf())).unwrap();
            let prompt = crate::open_responses::render::render_qwen_serve_prompt(&request);
            let ids = tokenizer.encode(&prompt, false).unwrap();
            (ids.len() == width).then_some(ids)
        }).expect("valid rendered prompt at requested width");
        eprintln!(
            "fresh-pilot width={width} prompt_sha256={}",
            token_ids_sha256_i32le(&tokens)
        );
        let mut states = Vec::new();
        let mut outputs = Vec::new();
        let mut costs = Vec::new();
        for packed in [false, true] {
            let plan = packed.then(|| {
                qwen_llm::metal_dflash::plan_single_chunk_prefill_scratch(
                    loaded.metal_model(),
                    width as u32,
                    width,
                    PrefillScratchConfig::default(),
                )
                .unwrap()
            });
            let price = plan.as_ref().map_or(0, |p| {
                p.priced_upper_bound(|bytes| {
                    Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size)
                })
                .unwrap()
            });
            assert!(price <= 128 * 1024 * 1024);
            assert!(
                loaded
                    .qwen_execution_memory_admission(1, width + 64, price, 0)
                    .unwrap()
                    .admitted
            );
            let before = loaded.context().current_allocated_size();
            let start = Instant::now();
            let mut scratch = plan.map(|plan| {
                MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
                    loaded.context(),
                    loaded.metal_model(),
                    plan,
                )
                .unwrap()
            });
            let scratch_bytes = loaded.context().current_allocated_size() - before;
            assert!(scratch_bytes <= 128 * 1024 * 1024);
            let mut sequence = loaded
                .create_sequence(SequenceConfig::new(width + 64))
                .unwrap();
            let allocation_ms = start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            let logits = if packed {
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
                .unwrap_or_else(|_| panic!("fresh prefill failed"))
                .unwrap()
            } else {
                // Explicit token-by-token reference: serve now chunks fresh
                // prompts over the serial-tail limit, so this arm must not go
                // through the policy.
                let mut last = Vec::new();
                for (position, &token) in tokens.iter().enumerate() {
                    last = forward
                        .single_token(token, position as u32, unsafe {
                            sequence.metal_session_mut()
                        })
                        .expect("fresh serial prefill");
                    sequence.advance_by(1).unwrap();
                }
                last
            };
            let prefill_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(sequence.position(), width);
            eprintln!(
                "fresh-pilot width={width} packed={packed} priced_bytes={price} scratch_bytes={scratch_bytes} allocation_ms={allocation_ms:.3} prefill_ms={prefill_ms:.3}"
            );
            costs.push(allocation_ms + prefill_ms);
            states.push(sequence);
            outputs.push(logits);
        }
        let logits_cos = cosine(&outputs[0], &outputs[1]);
        let snapshots: Vec<_> = states
            .iter()
            .map(|s| {
                s.metal_session()
                    .snapshot(loaded.snapshot_identity(s).unwrap(), tokens.clone(), None)
                    .unwrap()
            })
            .collect();
        let state_cos = consumed_tail_state_cosine(&snapshots[0], &snapshots[1], 0);
        drop(snapshots);
        assert!(logits_cos >= 0.999 && state_cos >= 0.999);
        let request = ServeRequest {
            temperature: Some(0.0),
            ..ServeRequest::default()
        };
        let mut sampler = request_sampler(&request).unwrap();
        let mut generated = Vec::new();
        for step in 0..64 {
            let next: Vec<_> = outputs
                .iter()
                .map(|logits| sampler.sample(logits).unwrap().token)
                .collect();
            assert_eq!(next[0], next[1], "width={width} step={step}");
            generated.push(next[0]);
            if stops.contains(&next[0]) || step == 63 {
                break;
            }
            for (state, logits) in states.iter_mut().zip(&mut outputs) {
                *logits = forward
                    .single_token(next[0], state.position() as u32, unsafe {
                        state.metal_session_mut()
                    })
                    .unwrap();
                state.advance_by(1).unwrap();
            }
        }
        let saving = 100.0 * (1.0 - costs[1] / costs[0]);
        eprintln!(
            "fresh-pilot width={width} logits_cos={logits_cos:.10} min_state_cos={state_cos:.10} greedy_tokens={} phase_saving_pct={saving:.3}",
            generated.len()
        );
        assert!(
            saving >= 25.0,
            "fresh packed allocation+prefill misses phase gate"
        );
    }
}

#[test]
#[ignore = "CPU fixture builder; requires QWEN_FRESH_PILOT_MODEL and QWEN_FRESH_HTTP_OUT"]
fn fresh_packed_endpoint_fixture() {
    use std::io::Write;
    let model = std::env::var("QWEN_FRESH_PILOT_MODEL").unwrap();
    let gguf = GgufFile::open(&model).unwrap();
    let family = qwen_llm::model_family::ModelFamily::detect(&gguf).unwrap();
    let template = crate::prompt_template::serve_qwen_template(family, &gguf).unwrap();
    let tokenizer = Tokenizer::open(&model).unwrap();
    let mut cases = Vec::new();
    for (name, width, text, maximum, temperature) in [
        (
            "fresh-19",
            19,
            "Reply with exactly the word violet.",
            16,
            0.0,
        ),
        (
            "exact-19",
            19,
            "Reply with exactly the word violet.",
            16,
            0.0,
        ),
        (
            "fresh-32-code",
            32,
            "Write Python that merges two sorted lists without duplicates.",
            128,
            0.0,
        ),
        (
            "fresh-48-prose",
            48,
            "Explain how a binary search tree stores and finds values.",
            128,
            0.0,
        ),
        ("guard-18", 18, "Reply with the word amber.", 16, 0.0),
        (
            "guard-49",
            49,
            "Write Python that computes the greatest common divisor.",
            128,
            0.0,
        ),
        (
            "sampled-32",
            32,
            "Reply with exactly the word green.",
            16,
            0.7,
        ),
        (
            "guard-512",
            512,
            "Reply with exactly the word indigo.",
            16,
            0.0,
        ),
    ] {
        let (body, ids) = (0..1024).find_map(|padding| {
            let body = serde_json::json!({"model": "fresh-pilot", "input": format!("{}{text}", "Hello ".repeat(padding)),
                "max_output_tokens": maximum, "temperature": temperature,
                "x_qwen": {"no_thinking": true, "seed": 1729}});
            let request = crate::open_responses::items::parse_request(&body).unwrap();
            let bound = crate::open_responses::bind_qwen_request(&request, template,
                crate::supports_qwen_no_thinking_prompt(family, &gguf)).unwrap();
            let prompt = crate::open_responses::render::render_qwen_serve_prompt(&bound);
            let ids = tokenizer.encode(&prompt, false).unwrap();
            (ids.len() == width).then_some((body, ids))
        }).expect("full rendered request at intended token count");
        eprintln!(
            "fresh-fixture name={name} tokens={width} sha256={}",
            token_ids_sha256_i32le(&ids)
        );
        cases.push(serde_json::json!({"name": name, "request": body}));
    }
    let path = std::env::var("QWEN_FRESH_HTTP_OUT").unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&cases).unwrap())
        .unwrap();
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
            // Explicit token-by-token reference: the serve policy chunks a
            // 32-token remainder, so this arm must not go through it.
            let mut last = Vec::new();
            for (offset, &token) in tokens[prefix..].iter().enumerate() {
                last = forward
                    .single_token(token, (prefix + offset) as u32, unsafe {
                        sequence.metal_session_mut()
                    })
                    .expect("serial prefill");
                sequence.advance_by(1).unwrap();
            }
            last
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
    // Packed is bitwise equal to the general chunked plan (asserted above);
    // this bounds chunked-vs-token-by-token drift in persistent state. The
    // original 0.999 was set on Q8; Q4 weights drift further on every
    // chunked prefill, not just packed tails (2026-09-23, 11.3K+32: 3.8 Q4
    // 0.9984, 3.6 Q4 0.9994, logits cos >= 0.9999993).
    assert!(min_cos >= 0.998);
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
