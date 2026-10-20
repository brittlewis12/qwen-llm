use super::*;

#[test]
#[ignore = "GLM53_GGUF + GLM53_DRAFTER_GGUF, integrated policy/API, GPU lease and MTL_DEBUG_LAYER=1"]
fn gpu_resident_drafter_cold_prefill_repeat_and_cancel() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let mut invocation = invocation(&path, Some(4096), Some(12));
    invocation.drafter = Some(std::env::var("GLM53_DRAFTER_GGUF").unwrap().into());
    let prepared = Prepared::new(&source, &invocation).unwrap();
    let ctx = MetalContext::new().unwrap();
    let head = dflash::load_head(&ctx, prepared.drafter.as_ref().unwrap()).unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let cache = crate::serve::SnapshotCachePlan::resolve(
        Some(64),
        Default::default(),
        ctx.memory_signals(),
    )
    .unwrap();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache,
    );
    backend.drafter = Some(head);
    assert!(
        backend
            .describe_snapshots()
            .contains("dflash_mode=cold_prefill")
    );
    assert_eq!(backend.lineage, PackedLineage::Fast);
    let request = prepared_request(
        &backend,
        json!({
            "model": "glm", "input": "Count from one to twenty.",
            "temperature": 0, "max_output_tokens": 12, "reasoning": {"effort": "low"}
        }),
    );
    let mut expected = None;
    for _ in 0..2 {
        let mut sink = Sink::default();
        let outcome = backend
            .generate_prepared(Arc::clone(&request), &mut sink)
            .unwrap();
        assert_eq!(outcome.usage.cached_tokens, 0);
        assert_eq!(outcome.stats.unwrap().matched_tokens, 0);
        assert!(outcome.usage.output_tokens > 0);
        assert!(backend.session.is_none());
        assert!(backend.history.is_empty());
        assert!(backend.chunk_origin.is_none());
        assert_eq!(backend.cache.len(), 0);
        if let Some((bytes, count, end)) = &expected {
            assert_eq!(&sink.bytes, bytes);
            assert_eq!(outcome.usage.output_tokens, *count);
            assert_eq!(&outcome.end, end);
        } else {
            expected = Some((sink.bytes, outcome.usage.output_tokens, outcome.end));
        }
    }
    for mut sink in [
        Sink {
            abort_tick: Some(3),
            ..Default::default()
        },
        Sink {
            abort_piece: Some(2),
            ..Default::default()
        },
    ] {
        assert!(matches!(
            backend.generate_prepared(Arc::clone(&request), &mut sink),
            Err(BackendFailure::Aborted(_))
        ));
        assert!(backend.session.is_none());
        assert!(backend.history.is_empty());
        assert!(backend.chunk_origin.is_none());
        assert_eq!(backend.cache.len(), 0);
    }
    let mut sink = Sink::default();
    let recovered = backend.generate_prepared(request, &mut sink).unwrap();
    let (bytes, count, end) = expected.unwrap();
    assert_eq!(sink.bytes, bytes);
    assert_eq!(recovered.usage.output_tokens, count);
    assert_eq!(recovered.end, end);
    assert_eq!(recovered.usage.cached_tokens, 0);
}
