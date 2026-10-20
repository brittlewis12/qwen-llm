use super::*;

#[test]
fn borrowed_draft_api_keeps_target_and_session_lifetimes_independent() {
    fn borrow<'a, 's>(
        ctx: &'a MetalContext,
        embedding: &'a MetalTensor,
        output: &'a MetalTensor,
        head: &'a MetalDFlashHead,
        session: &'s mut MetalDFlashSession,
        hidden_size: u32,
        vocab_size: u32,
    ) -> MetalDFlashDraftDecoder<'a, 's> {
        MetalDFlashDraftDecoder::new(
            MetalDFlashDraftTarget {
                ctx,
                hidden_size,
                vocab_size,
                token_embedding: embedding,
                output,
            },
            head,
            session,
        )
    }

    // Type-check the adapter surface without creating a Metal device/session.
    let _ = borrow;
    let _: fn(&mut MetalDFlashDraftDecoder<'_, '_>, i32, u32) -> Result<Vec<i32>, DFlashError> =
        |decoder, carry, pos| decoder.draft_block(carry, pos);
    let _: fn(
        &mut MetalDFlashDraftDecoder<'_, '_>,
        i32,
        u32,
        &mut Sampler,
    ) -> Result<DFlash2SparseProposalBlock, DFlashError> =
        |decoder, carry, pos, sampler| decoder.draft_block_sampled(carry, pos, sampler);
    let _: fn(
        &mut MetalDFlashDraftDecoder<'_, '_>,
        i32,
        u32,
    ) -> Result<DFlash2SelectorDiagnostic, DFlashError> =
        |decoder, carry, pos| decoder.draft_block_with_selector_diagnostic(carry, pos);
    let _: fn(&mut MetalDFlashDraftDecoder<'_, '_>, i32, u32) -> Result<Vec<f32>, DFlashError> =
        |decoder, carry, pos| decoder.draft_block_with_logits(carry, pos);
}

#[test]
fn draft_session_geometry_uses_target_dimensions_and_capture_count() {
    let cfg = crate::loader::DFlashConfig {
        n_layer: 2,
        hidden_size: 8,
        intermediate_size: 16,
        n_q_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        causal: false,
        swa_window: 7,
        block_size: 4,
        mask_token_id: 0,
        n_target_features_layers: 3,
        conv_kernel_size: 0,
        conv_group_size: 0,
        selector_rank: 0,
        selector_top_k: 0,
    };
    let geometry = DFlashSessionGeometry::new(&cfg, 3, 8, 37, 11).unwrap();
    assert_eq!(geometry.n_target_features, 24);
    assert_eq!(geometry.logits_elems, 4 * 37);
    assert_eq!(geometry.x_elems, 4 * 8);
    assert_eq!(geometry.context_allocation_elements(2).unwrap()[0], 11 * 24);

    let resized = DFlashSessionGeometry::new(&cfg, 2, 16, 41, 11).unwrap();
    assert_eq!(resized.n_target_features, 32);
    assert_eq!(resized.logits_elems, 4 * 41);
    assert_eq!(resized.x_elems, geometry.x_elems);
    assert_eq!(resized.kv_ctx_elems, geometry.kv_ctx_elems);
}
