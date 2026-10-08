//! Map #15 gates for GLM-5.3 RAM snapshots: capture -> restore -> suffix
//! equals the uninterrupted run with the same segmentation, bit for bit in
//! logits and captured end state, in both lineages, at every pool residue,
//! around the sparse frontier, at a cancelled prefill's committed boundary,
//! into a recycled destination of another capacity; and every refusal
//! writes nothing.

use super::*;

const SUFFIX: usize = 37;
const STEPS: usize = 4;

/// Logits of `suffix` prefilled after `prefix` state, then `steps` decode
/// steps teacher-forced from `tokens`.
fn continue_from(
    ctx: &MetalContext,
    session: &mut Glm5NextSession<'_>,
    tokens: &[u32],
    from: usize,
) -> Vec<Vec<f32>> {
    let mut logits = vec![
        session
            .prefill_packed(ctx, &tokens[from..from + SUFFIX])
            .unwrap(),
    ];
    for &token in &tokens[from + SUFFIX..from + SUFFIX + STEPS] {
        logits.push(session.forward(ctx, token).unwrap());
    }
    logits
}

fn fresh<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    lineage: PackedLineage,
) -> Glm5NextSession<'w> {
    let mut s =
        Glm5NextSession::with_prefill_rows(ctx, weights, capacity, 512.min(capacity)).unwrap();
    s.set_packed_lineage(lineage);
    s
}

#[test]
#[ignore = "map #15 snapshot gates: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn snapshot_restore_equals_the_uninterrupted_run() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let frontier = weights.config.sparse_frontier() as usize;
    let mut failures = Vec::new();
    let cases: [(PackedLineage, Vec<usize>); 2] = [
        (
            PackedLineage::Exact,
            vec![97, 98, 99, 100, frontier - 1, frontier],
        ),
        (
            PackedLineage::Fast,
            (97..=100).chain(frontier - 2..=frontier + 2).collect(),
        ),
    ];
    for (lineage, positions) in cases {
        for n in positions {
            // A: uninterrupted, same segmentation (prefix, then suffix).
            let mut a = fresh(&ctx, &weights, n + SUFFIX + STEPS + 8, lineage);
            a.prefill_packed(&ctx, &tokens[..n]).unwrap();
            let a_logits = continue_from(&ctx, &mut a, &tokens, n);
            let a_end = a.capture_snapshot().unwrap();
            drop(a);
            // B: capture at n, restore into a recycled session of another
            // capacity that first ran unrelated tokens.
            let mut source = fresh(&ctx, &weights, n + 1, lineage);
            source.prefill_packed(&ctx, &tokens[..n]).unwrap();
            let snapshot = source.capture_snapshot().unwrap();
            assert_eq!(snapshot.position(), n);
            assert_eq!(
                snapshot.bytes() as u64,
                snapshot_bytes(&weights.config, n as u64).unwrap(),
                "byte estimator"
            );
            drop(source);
            let mut b = fresh(&ctx, &weights, n + SUFFIX + STEPS + 72, lineage);
            b.prefill_packed(&ctx, &tokens[3000..3000 + 61]).unwrap();
            b.restore_snapshot(&snapshot).unwrap();
            assert_eq!(b.position(), n);
            let b_logits = continue_from(&ctx, &mut b, &tokens, n);
            let b_end = b.capture_snapshot().unwrap();
            let logits_equal = logit_bits(&a_logits) == logit_bits(&b_logits);
            let state_equal = a_end.same_state(&b_end);
            eprintln!(
                "{lineage:?} n={n} (n % pool = {}): logits bitwise {logits_equal}, end state equal {state_equal}, snapshot {} bytes",
                n % weights.config.indexer_pool as usize,
                snapshot.bytes()
            );
            if !(logits_equal && state_equal) {
                failures.push(format!(
                    "{lineage:?} n={n}: logits {logits_equal}, end state {state_equal}"
                ));
            }
        }
    }

    // A prefill cancelled at its second chunk boundary commits 512 tokens;
    // its snapshot continues exactly as an uninterrupted 512 + suffix run.
    let lineage = PackedLineage::Fast;
    let mut source = fresh(&ctx, &weights, 1300 + 1, lineage);
    let mut calls = 0;
    let cancelled = source.prefill_packed_with_checkpoint(&ctx, &tokens[..1300], &mut || {
        calls += 1;
        if calls == 2 {
            Err("cancel".into())
        } else {
            Ok(())
        }
    });
    assert!(matches!(cancelled, Err(Glm5NextMetalError::Cancelled(_))));
    assert_eq!(source.position(), 512);
    let snapshot = source.capture_snapshot().unwrap();
    drop(source);
    let mut a = fresh(&ctx, &weights, 512 + SUFFIX + STEPS + 8, lineage);
    a.prefill_packed(&ctx, &tokens[..512]).unwrap();
    let a_logits = continue_from(&ctx, &mut a, &tokens, 512);
    drop(a);
    let mut b = fresh(&ctx, &weights, 512 + SUFFIX + STEPS + 8, lineage);
    b.restore_snapshot(&snapshot).unwrap();
    let b_logits = continue_from(&ctx, &mut b, &tokens, 512);
    if logit_bits(&a_logits) != logit_bits(&b_logits) {
        failures.push("cancelled boundary: logits differ".into());
    }

    // Refusals write nothing: the destination continues as before.
    let mut exact = fresh(&ctx, &weights, 600, PackedLineage::Exact);
    exact.prefill_packed(&ctx, &tokens[..100]).unwrap();
    let before = exact.capture_snapshot().unwrap();
    let refused =
        |result: Result<()>| matches!(result, Err(Glm5NextMetalError::SnapshotMismatch(_)));
    assert!(refused(exact.restore_snapshot(&snapshot)), "lineage");
    let mut forged = exact.capture_snapshot().unwrap();
    forged.forge_identity(u64::MAX, SNAPSHOT_POLICY_VERSION);
    assert!(refused(exact.restore_snapshot(&forged)), "weights instance");
    let mut forged = exact.capture_snapshot().unwrap();
    forged.forge_identity(weights.instance, SNAPSHOT_POLICY_VERSION + 1);
    assert!(refused(exact.restore_snapshot(&forged)), "policy version");
    let mut small = fresh(&ctx, &weights, 100, PackedLineage::Exact);
    assert!(
        refused(small.restore_snapshot(&before)),
        "no room past the position"
    );
    assert_eq!(exact.position(), 100);
    assert!(
        exact.capture_snapshot().unwrap().same_state(&before),
        "a refusal wrote state"
    );

    // A poisoned session cannot be captured.
    let mut poisoned = fresh(&ctx, &weights, frontier + 64, PackedLineage::Fast);
    poisoned.corrupt_sparse_row = Some(0);
    assert!(
        poisoned
            .prefill_packed(&ctx, &tokens[..frontier + 8])
            .is_err()
    );
    assert!(matches!(
        poisoned.capture_snapshot(),
        Err(Glm5NextMetalError::Poisoned)
    ));

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
