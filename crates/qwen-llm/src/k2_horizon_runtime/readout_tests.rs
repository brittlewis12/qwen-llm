use super::*;

thread_local! {
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
pub(super) fn record_head() {
    COUNTS.with(|c| {
        let (heads, downloads) = c.get();
        c.set((heads + 1, downloads));
    });
}
pub(super) fn record_download() {
    COUNTS.with(|c| {
        let (heads, downloads) = c.get();
        c.set((heads, downloads + 1));
    });
}
fn take_counts() -> (usize, usize) {
    COUNTS.with(|c| c.replace((0, 0)))
}

fn cache_bytes(session: &K2Session<'_, '_>) -> Vec<u8> {
    let cache = &session.buffers.cache;
    unsafe {
        std::slice::from_raw_parts(
            cache.buffer.contents().as_ptr().cast::<u8>(),
            cache.n_bytes() as usize,
        )
    }
    .to_vec()
}

fn same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
    );
}

#[test]
#[ignore = "K2_GGUF production lease/API validation; advance/append exactness, counts and poison semantics"]
fn gpu_k2_advance_skips_readouts_preserves_state_and_poisoning() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").unwrap()).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut model = K2LoadedModel::load(&ctx, &source, 72).unwrap();
    let tokens: Vec<u32> = (0..67).map(|i| [0, 42, 17, 91, 1024, 222][i % 6]).collect();
    for (mode, chunk) in [
        (PrefillMode::Serial, 1),
        (PrefillMode::BatchQ8, 32),
        (PrefillMode::BatchQ8, 17),
    ] {
        model.prefill = mode;
        let mut session = model.create_session(0).unwrap();
        take_counts();
        let mut chunks = tokens.chunks(chunk).peekable();
        let mut baseline = None;
        while let Some(ids) = chunks.next() {
            if chunks.peek().is_none() {
                baseline = Some(session.append_with_captures(ids, &[0, 17, 35]).unwrap());
            } else {
                session.append(ids).unwrap();
            }
        }
        let baseline = baseline.unwrap();
        assert_eq!(
            take_counts(),
            (tokens.len().div_ceil(chunk), tokens.len().div_ceil(chunk))
        );
        let baseline_cache = cache_bytes(&session);
        let continuation = session.append(&[3, 19]).unwrap();
        let continued_cache = cache_bytes(&session);
        drop(session);

        let mut session = model.create_session(0).unwrap();
        assert!(session.advance(&[]).is_err());
        assert!(session.advance(&[0, model.config().vocab_size]).is_err());
        assert!(!session.is_poisoned());
        assert_eq!(session.committed_len(), 0);
        // Stale logits must neither be scanned nor returned by advancement.
        unsafe {
            session
                .buffers
                .logits
                .buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .write(f32::NAN)
        };
        take_counts();
        let mut chunks = tokens.chunks(chunk).peekable();
        let mut candidate = None;
        while let Some(ids) = chunks.next() {
            if chunks.peek().is_none() {
                candidate = Some(session.append_with_captures(ids, &[0, 17, 35]).unwrap());
            } else {
                session.advance(ids).unwrap();
                assert_eq!(take_counts(), (0, 0));
            }
        }
        let candidate = candidate.unwrap();
        assert_eq!(take_counts(), (1, 1));
        same_bits(&candidate.logits, &baseline.logits);
        same_bits(&candidate.residuals, &baseline.residuals);
        assert_eq!(cache_bytes(&session), baseline_cache);
        assert_eq!(session.committed_len(), 67);
        same_bits(&session.append(&[3, 19]).unwrap(), &continuation);
        assert_eq!(cache_bytes(&session), continued_cache);
        drop(session);

        let mut session = model.create_session(0).unwrap();
        session.fail_next_append_check = true;
        take_counts();
        assert!(session.advance(&tokens[..chunk]).is_err());
        assert_eq!(take_counts(), (0, 0));
        assert!(session.is_poisoned());
        assert_eq!(session.committed_len(), 0);
        assert!(session.advance(&[0]).is_err());
        assert!(session.append(&[0]).is_err());
        drop(session);
        drop(model.create_session(0).unwrap());
    }
}

#[test]
#[ignore = "K2_GGUF paired diagnostic, not product benchmark; loaded model/fresh sessions, same token/chunk policy"]
fn gpu_k2_readout_schedule_paired_wall_probe() {
    use std::time::Instant;
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").unwrap()).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut model = K2LoadedModel::load(&ctx, &source, 128).unwrap();
    let tokens: Vec<u32> = (0..128)
        .map(|i| [0, 42, 17, 91, 1024, 222][i % 6])
        .collect();
    let mut rows = Vec::new();
    for (mode, chunk) in [(PrefillMode::Serial, 1), (PrefillMode::BatchQ8, 32)] {
        model.prefill = mode;
        let mut reference: Option<Vec<f32>> = None;
        // Warm both policies, then alternate paired order to avoid one-way warmup bias.
        for repetition in 0..4 {
            for advance in if repetition % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mut session = model.create_session(0).unwrap();
                take_counts();
                let start = Instant::now();
                let mut logits = Vec::new();
                let mut chunks = tokens.chunks(chunk).peekable();
                while let Some(ids) = chunks.next() {
                    if advance && chunks.peek().is_some() {
                        session.advance(ids).unwrap();
                    } else {
                        logits = session.append(ids).unwrap();
                    }
                }
                let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                let counts = take_counts();
                let expected = if advance {
                    1
                } else {
                    tokens.len().div_ceil(chunk)
                };
                assert_eq!(counts, (expected, expected));
                if let Some(reference) = &reference {
                    same_bits(&logits, reference);
                } else {
                    reference = Some(logits);
                }
                assert_eq!(session.committed_len(), 128);
                rows.push(serde_json::json!({"mode":format!("{mode:?}"), "chunk_tokens":chunk,
                    "repetition":repetition,"warmup":repetition==0,"advance_nonfinal_chunks":advance,
                    "prefill_wall_ms":wall_ms,"head_encodes":counts.0,"logits_downloads":counts.1}));
            }
        }
    }
    let report = serde_json::json!({"status":"passed", "scope":"paired_loaded_model_synthetic_prefill_diagnostic_not_end_to_end_product_benchmark", "prompt_tokens":128,"metal_debug_layer":true,"samples":rows});
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Ok(path) = std::env::var("K2_READOUT_EVIDENCE") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
