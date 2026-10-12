//! Direct backend cancellation coverage for the unforced production policy.
use super::*;

fn suffix_fixture(
    backend: &Glm5NextBackend<'_, '_>,
    suffix: usize,
) -> (Arc<PreparedResponse>, Vec<u32>, usize) {
    // One-token words make exact CPU sizing inexpensive. This is a state-machine
    // fixture, not a quality prompt or performance packet.
    let mut words = suffix.saturating_sub(32);
    for _ in 0..8 {
        let prompt = prepared_request(
            backend,
            json!({
                "model":"glm","instructions":"Answer briefly.",
                "input":[{"role":"user","content":format!("Context:{}\nWhat is 3 + 4? Reply with the number only.", " word".repeat(words))}],
                "reasoning":{"effort":"low"},"temperature":0,"max_output_tokens":32
            }),
        );
        let encode = |text: &str| {
            decode_loop::encode_checked(
                backend.prepared.artifact.tokenizer(),
                text,
                false,
                backend.prepared.artifact.config().vocab_size,
                FAMILY,
            )
        };
        let tokens = encode(&prompt.prompt).unwrap();
        let cuts = cut_positions(
            encode,
            &prompt.prompt,
            &tokens,
            prompt.boundaries,
            Schedule::FastSharedSplitV1,
        );
        assert_eq!(cuts.len(), 1);
        let actual = tokens.len() - cuts[0];
        if actual == suffix {
            return (prompt, tokens, cuts[0]);
        }
        words = if actual > suffix {
            words.checked_sub(actual - suffix).unwrap()
        } else {
            words.checked_add(suffix - actual).unwrap()
        };
    }
    panic!("tokenizer did not size the bounded suffix fixture");
}

fn run(
    backend: &mut Glm5NextBackend<'_, '_>,
    request: &Arc<PreparedResponse>,
) -> (GenerationOutcome, Vec<u32>, Vec<u8>) {
    let mut sink = Sink::default();
    let outcome = backend
        .generate_prepared(Arc::clone(request), &mut sink)
        .unwrap();
    (outcome, backend.last_prefill_logits.clone(), sink.bytes)
}

fn equal(
    reference: &(GenerationOutcome, Vec<u32>, Vec<u8>),
    candidate: &(GenerationOutcome, Vec<u32>, Vec<u8>),
) {
    assert!(
        candidate
            .1
            .iter()
            .all(|bits| f32::from_bits(*bits).is_finite())
    );
    assert_eq!(candidate.1, reference.1, "full prefill logits");
    assert_eq!(candidate.2, reference.2, "all emitted greedy bytes");
    assert_eq!(candidate.0.end, reference.0.end);
    assert_eq!(
        candidate.0.usage.output_tokens,
        reference.0.usage.output_tokens
    );
}

#[test]
#[ignore = "GLM53_GGUF bounded default Fast Half 2048-row checkpoint cancel/resume; production lease; no server"]
fn gpu_chunk_checkpoint_cancel_resume() {
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let invocation = invocation(&path, Some(8192), Some(32));
    let prepared = Prepared::new(&source, &invocation).unwrap();
    assert_eq!(prepared.fast_precision, FastPrecision::Half);
    let ctx = MetalContext::new().unwrap(); // Production library owns the lease.
    let weights = load(&ctx, &source, &prepared).unwrap();
    let cache_plan =
        crate::serve::SnapshotCachePlan::resolve(None, Default::default(), ctx.memory_signals())
            .unwrap();
    let mut incumbent = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache_plan,
    );
    assert!(incumbent.prefix_reuse && incumbent.fast_snapshots);
    assert_eq!(incumbent.lineage, PackedLineage::Fast);
    assert_eq!(incumbent.chunk_checkpoint_override, None);
    incumbent.chunk_checkpoint_override = Some(false);
    let (prompt, tokens, shared_cut) = suffix_fixture(&incumbent, 4099);
    assert!(shared_cut > 0 && shared_cut < 2048);
    let reference = run(&mut incumbent, &prompt);
    assert_eq!(
        incumbent.frozen_rows,
        Some(2048),
        "test needs admitted R2048, not forced workspace"
    );
    assert_eq!(reference.0.usage.cached_tokens, 0);
    drop(incumbent);

    let prepared = Prepared::new(&source, &invocation).unwrap();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache_plan,
    );
    assert_eq!(backend.chunk_checkpoint_override, None);
    assert!(backend.chunk_checkpoints_allowed() && backend.prefix_reuse && backend.fast_snapshots);
    assert_eq!(backend.lineage, PackedLineage::Fast);
    let committed = shared_cut + 2048;
    let target = shared_cut + 4096;
    assert_eq!(
        chunk_checkpoint::target(tokens.len(), shared_cut, 2048),
        Some(target)
    );
    // Admission, shared segment, first full suffix chunk, then abort BEFORE
    // the second suffix chunk. The last-full-chunk target has not been reached.
    let mut cancelled = Sink {
        abort_tick: Some(4),
        ..Sink::default()
    };
    assert!(matches!(
        backend.generate_prepared(Arc::clone(&prompt), &mut cancelled),
        Err(BackendFailure::Aborted(_))
    ));
    assert_eq!(backend.frozen_rows, Some(2048));
    let session = backend.session.as_ref().unwrap();
    assert!(!session.is_poisoned());
    assert_eq!(session.position(), committed);
    assert_eq!(backend.history, tokens[..committed]);
    assert!(backend.chunk_origin.unwrap().matches(
        shared_cut,
        2048,
        FastPrecision::Half,
        committed
    ));
    assert!(
        !backend.live_canonical,
        "off-grid shared cut is not the from-zero schedule"
    );
    let shared_ns = CacheNamespace {
        lineage: PackedLineage::Fast,
        schedule: Schedule::FastSharedSplitV1,
        rows: 2048,
    };
    let deep_ns = chunk_checkpoint::namespace(shared_ns, shared_cut, FastPrecision::Half).unwrap();
    assert!(
        backend
            .cache
            .entry_for_in(&shared_ns, &tokens[..shared_cut])
            .is_some(),
        "correctness fixture needs admitted shared capture"
    );
    assert!(
        backend
            .cache
            .entry_for_in(&deep_ns, &tokens[..target])
            .is_none()
    );

    let resumed = run(&mut backend, &prompt);
    equal(&reference, &resumed);
    assert_eq!(resumed.0.usage.cached_tokens, committed);
    let phases = backend.last_chunk_checkpoint.as_ref().unwrap();
    assert_eq!(phases["reuse_source"], "live");
    assert_eq!(phases["restore_origin"], "none");
    assert_eq!(
        phases["chunk_captures"],
        json!([format!("{target}:captured")])
    );
    assert!(
        backend
            .cache
            .entry_for_in(&deep_ns, &tokens[..target])
            .is_some(),
        "resume must publish the checkpoint, not silently fall back in this correctness fixture"
    );
    assert!(
        backend.chunk_origin.is_none(),
        "decode invalidates live chunk provenance"
    );
    assert_eq!(
        backend.session.as_ref().unwrap().packed_lineage(),
        Some(PackedLineage::Fast)
    );
    assert_eq!(
        backend.session.as_ref().unwrap().fast_precision(),
        Some(FastPrecision::Half)
    );

    let restored = run(&mut backend, &prompt);
    equal(&reference, &restored);
    assert_eq!(restored.0.usage.cached_tokens, target);
    let phases = backend.last_chunk_checkpoint.as_ref().unwrap();
    assert_eq!(phases["restore_origin"], "fast_chunk_checkpoint_v1");
    assert_eq!(phases["execution_schedule"], "fast_shared_split_v1");
    assert_eq!(phases["prefill_tokens"], 3);
    assert_eq!(backend.chunk_checkpoint_override, None);
    assert_eq!(backend.cache.pinned_bytes(), 0);
    assert!(backend.chunk_origin.is_none());
    eprintln!(
        "default_chunk_checkpoint_cancel_resume: shared={shared_cut} committed={committed} checkpoint={target} prompt={} full logits/greedy output/end equal",
        tokens.len()
    );
    drop(backend);

    // Cheap actual-backend denial coverage, separately labelled: cap this tiny
    // fixture's workspace at 8 rows rather than replay another 4K prompt. The
    // policy is still unforced and the planner must not shrink/change the width
    // in response to a denied optional capture. This is not a timing cell.
    let small_invocation = super::invocation(&path, Some(256), Some(32));
    let small_prepared = || {
        let mut prepared = Prepared::new(&source, &small_invocation).unwrap();
        prepared.prefill_rows = 8;
        prepared
    };
    let mut incumbent = Glm5NextBackend::new(
        &ctx,
        &weights,
        small_prepared(),
        "glm".into(),
        std::time::Duration::ZERO,
        cache_plan,
    );
    incumbent.chunk_checkpoint_override = Some(false);
    let (small, small_tokens, small_shared) = suffix_fixture(&incumbent, 65);
    let small_reference = run(&mut incumbent, &small);
    assert_eq!(incumbent.frozen_rows, Some(8));
    drop(incumbent);
    let mut denied = Glm5NextBackend::new(
        &ctx,
        &weights,
        small_prepared(),
        "glm".into(),
        std::time::Duration::ZERO,
        cache_plan,
    );
    assert_eq!(denied.chunk_checkpoint_override, None);
    denied.deny_captures = true;
    let refused = run(&mut denied, &small);
    equal(&small_reference, &refused);
    let small_target = chunk_checkpoint::target(small_tokens.len(), small_shared, 8).unwrap();
    assert_eq!(
        denied.last_chunk_checkpoint.as_ref().unwrap()["chunk_captures"],
        json!([format!("{small_target}:denied")])
    );
    assert_eq!(denied.frozen_rows, Some(8));
    assert_eq!(denied.cache.len(), 0);
    assert_eq!(denied.cache.pinned_bytes(), 0);
    assert_eq!(
        denied.session.as_ref().unwrap().packed_lineage(),
        Some(PackedLineage::Fast)
    );
    assert_eq!(
        denied.session.as_ref().unwrap().fast_precision(),
        Some(FastPrecision::Half)
    );
    assert!(denied.chunk_origin.is_none());
    eprintln!(
        "default_chunk_checkpoint_denied: test workspace cap=8, suffix=65, optional deep capture denied, full logits/greedy output/end equal"
    );
}
