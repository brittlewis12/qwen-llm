//! Reset reuses storage but must behave like an independently fresh session.

use super::*;

fn fresh<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    rows: usize,
) -> Glm5NextSession<'w> {
    let mut session = Glm5NextSession::with_prefill_rows(ctx, weights, capacity, rows).unwrap();
    session.set_packed_lineage(PackedLineage::Fast).unwrap();
    session.set_fast_precision(FastPrecision::Half).unwrap();
    session
}

fn retained_pointers(session: &Glm5NextSession<'_>) -> Vec<(usize, u64, u64)> {
    let mut pointers = Vec::new();
    let mut record = |t: &MetalTensor| {
        pointers.push((t.buffer.contents().as_ptr() as usize, t.offset, t.n_bytes()));
    };
    for layer in &session.layers {
        match layer {
            LayerState::Kda { conv, state } => {
                record(conv);
                record(state);
            }
            LayerState::Mla {
                latent,
                pending,
                pooled,
            } => {
                record(latent);
                record(pending);
                record(pooled);
            }
        }
    }
    for route in session.routes.iter().flatten() {
        record(&route.ids);
        record(&route.weights);
        record(&route.status);
    }
    macro_rules! record_fields {
        ($s:expr; $($field:ident),+ $(,)?) => {
            $(record(&$s.$field);)+
        };
    }
    let s = &session.s;
    record_fields!(s; token, embedding, quarter, no_sink, hc_partial_dots,
        hc_partial_sumsq, mixes, pre, post, comb, collapsed, normed, block_out,
        q, k, v, rank_a, raw_gate, raw_beta, rank_b, output_gate, kda_out,
        query_a, query_r, query, latent_raw, latent, query_latent, output_latent,
        heads_out, index_key, index_gate, dense_gate, dense_up, router,
        expert_inner, expert_out, routed, shared_swiglu, shared, final_hidden,
        final_normed, logits);
    for residual in &s.residual {
        record(residual);
    }
    if let Some(hc) = &s.generic_hc {
        record_fields!(hc; normalized, ones);
    }
    if let Some(sparse) = &session.sparse {
        record_fields!(sparse; index_query, index_query_f16, index_weights,
            scores, pool_ids, pool_counts, visible_pools, visible_rows,
            row_ids, row_counts, select_status, attention_partials,
            attention_partial_stats);
    }
    // Packed internals are private to their sibling module. Its exposed
    // route buffers are sentinels; the device counter also covers its scratch.
    if let Some(packed) = &session.packed {
        for (_, ids) in packed.route_ids() {
            record(ids);
        }
    }
    pointers
}

fn reset_retaining_storage(ctx: &MetalContext, session: &mut Glm5NextSession<'_>) {
    let pointers = retained_pointers(session);
    let ledger = *session.ledger();
    let capacity = session.capacity;
    let rows = session.packed.as_ref().map(|p| p.rows());
    let lineage = session.packed_lineage();
    let precision = session.fast_precision();
    let observed = session.observed_allocation_delta();
    let allocated = ctx.current_allocated_size();
    session.reset().unwrap();
    assert_eq!(
        ctx.current_allocated_size(),
        allocated,
        "reset allocated/freed Metal storage"
    );
    assert_eq!(retained_pointers(session), pointers);
    assert_eq!(*session.ledger(), ledger);
    assert_eq!(session.capacity, capacity);
    assert_eq!(session.packed.as_ref().map(|p| p.rows()), rows);
    assert_eq!(session.packed_lineage(), lineage);
    assert_eq!(session.fast_precision(), precision);
    assert_eq!(session.observed_allocation_delta(), observed);
    assert_eq!(session.position(), 0);
    assert!(!session.is_poisoned());
}

fn assert_same_state(actual: &Glm5NextSession<'_>, expected: &Glm5NextSession<'_>) {
    assert_eq!(actual.position(), expected.position());
    let snapshot = actual.capture_snapshot().unwrap();
    assert!(
        snapshot.same_state(&expected.capture_snapshot().unwrap()),
        "snapshot differs at position {}",
        actual.position()
    );
    assert!(
        snapshot.same_state(&actual.capture_snapshot().unwrap()),
        "repeated capture changed at position {}",
        actual.position()
    );
}

fn compare_schedule(
    ctx: &MetalContext,
    actual: &mut Glm5NextSession<'_>,
    expected: &mut Glm5NextSession<'_>,
    tokens: &[u32],
) {
    let a = actual.prefill_packed(ctx, tokens).unwrap();
    let b = expected.prefill_packed(ctx, tokens).unwrap();
    assert_comparable("reset prefill", &a, &b);
    assert_eq!(
        logit_bits(&[a]),
        logit_bits(&[b]),
        "prefill length {}",
        tokens.len()
    );
    assert_same_state(actual, expected);
    // Exercise the partial pending ring through a full pool boundary.
    for &token in tokens.iter().cycle().take(4) {
        let a = actual.forward(ctx, token).unwrap();
        let b = expected.forward(ctx, token).unwrap();
        assert_comparable("reset decode", &a, &b);
        assert_eq!(
            logit_bits(&[a]),
            logit_bits(&[b]),
            "decode position {}",
            actual.position()
        );
        assert_same_state(actual, expected);
    }
}

#[test]
#[ignore = "reset gates: loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn reset_equals_fresh_and_retains_storage() {
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
    assert_eq!(
        weights.config.indexer_pool, 4,
        "these cases target pool residues 1/3/0"
    );
    let old_len = frontier + 7;
    let capacity = old_len + 8;
    let new_start = 61;
    assert!(tokens.len() >= new_start + frontier + 2);
    let mut recycled = fresh(&ctx, &weights, capacity, 2048);
    recycled.prefill_packed(&ctx, &tokens[..old_len]).unwrap();
    let old = recycled.capture_snapshot().unwrap();
    assert_eq!(old.position(), old_len);

    // One expensive old prefill, replayed only as CPU snapshot copies. The
    // independent oracle never restores a snapshot or calls reset.
    for n in [1, 3, 4, frontier + 2] {
        recycled.restore_snapshot(&old).unwrap();
        let mut oracle = fresh(&ctx, &weights, capacity, 2048);
        reset_retaining_storage(&ctx, &mut recycled);
        assert_same_state(&recycled, &oracle);
        reset_retaining_storage(&ctx, &mut recycled);
        assert_same_state(&recycled, &oracle);
        compare_schedule(
            &ctx,
            &mut recycled,
            &mut oracle,
            &tokens[new_start..new_start + n],
        );
    }

    // Poison refusal must preserve even stale capacity beyond the live prefix.
    let before = recycled.capture_snapshot().unwrap();
    let bits = state_bits(&recycled);
    let pointers = retained_pointers(&recycled);
    let ledger = *recycled.ledger();
    let allocated = ctx.current_allocated_size();
    let position = recycled.position();
    recycled.poisoned = true;
    assert!(matches!(
        recycled.reset(),
        Err(Glm5NextMetalError::Poisoned)
    ));
    assert!(recycled.is_poisoned());
    assert_eq!(recycled.position(), position);
    assert_eq!(ctx.current_allocated_size(), allocated);
    assert_eq!(retained_pointers(&recycled), pointers);
    assert_eq!(*recycled.ledger(), ledger);
    assert_eq!(recycled.capacity, capacity);
    assert_eq!(recycled.packed.as_ref().unwrap().rows(), 2048);
    assert_eq!(recycled.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(recycled.fast_precision(), Some(FastPrecision::Half));
    assert_eq!(
        state_bits(&recycled),
        bits,
        "poison refusal wrote persistent storage"
    );
    recycled.poisoned = false; // Only to inspect the deliberately poisoned state.
    assert!(before.same_state(&recycled.capture_snapshot().unwrap()));
    drop(bits);
    drop(before);
    drop(old);
    drop(recycled);

    // No packed/sparse scratch: prefill_packed takes the serial fallback.
    let mut serial = fresh(&ctx, &weights, 16, 0);
    serial.prefill(&ctx, &tokens[..7]).unwrap();
    let mut oracle = fresh(&ctx, &weights, 16, 0);
    reset_retaining_storage(&ctx, &mut serial);
    assert_same_state(&serial, &oracle);
    reset_retaining_storage(&ctx, &mut serial);
    assert_same_state(&serial, &oracle);
    compare_schedule(
        &ctx,
        &mut serial,
        &mut oracle,
        &tokens[new_start..new_start + 3],
    );
}
