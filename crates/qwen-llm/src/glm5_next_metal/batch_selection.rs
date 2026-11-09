//! Independent score tables share a selector grid, never persistent state.
use super::*;

crate::env_flag!(
    /// Default-on M5 Fast/Half independent decode top-512 selector grid over two to four members. Zero restores per-member selection. Uses existing admitted packed scratch, identical integer selection and the global DSA selector rollback. Exact/F32 and ordinary prefill dispatch are unchanged.
    default_on enabled,
    "QWEN_GLM53_BATCH_SELECTION"
);

/// Repack scores with the largest member capacity as stride. A smaller first
/// workspace may not fit: fall back before encoding, without extra allocation.
pub(super) fn stride(ctx: &MetalContext, seqs: &[&Glm5NextSession<'_>]) -> Option<usize> {
    if projection::features() & projection::SELECT == 0
        || !(2..=4).contains(&seqs.len())
        || !crate::metal::glm_select_members_supported(ctx)
    {
        return None;
    }
    let c = &seqs[0].weights.config;
    if c.selected_pool_count() != 512
        || seqs
            .iter()
            .any(|s| s.position + 1 < c.sparse_frontier() as usize)
    {
        return None;
    }
    let sp = seqs[0].packed.as_ref()?.sparse.as_ref()?;
    if sp.microbatch < seqs.len() {
        return None;
    }
    let stride = seqs
        .iter()
        .map(|s| s.sparse.as_ref().map(|s| s.scores.shape[0] as usize))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    (stride.checked_mul(seqs.len())? as u64 <= sp.scores.n_elements()).then_some(stride)
}
fn status(sp: &PackedSparseScratch, mla: usize, count: usize) -> MetalTensor {
    sp.select_status
        .view_subrange((mla * sp.rows) as u64, vec![count as u64])
}
pub(super) fn prepare(seqs: &[&Glm5NextSession<'_>]) -> Result<()> {
    let c = &seqs[0].weights.config;
    let sp = seqs[0].packed.as_ref().unwrap().sparse.as_ref().unwrap();
    let n = seqs.len();
    write_i32(
        &sp.visible_pools.view_subrange(0, vec![n as u64]),
        &seqs
            .iter()
            .map(|s| ((s.position + 1) / c.indexer_pool as usize) as i32)
            .collect::<Vec<_>>(),
    )?;
    write_i32(
        &sp.visible_rows.view_subrange(0, vec![n as u64]),
        &seqs
            .iter()
            .map(|s| (s.position + 1) as i32)
            .collect::<Vec<_>>(),
    )?;
    for i in 0..c.block_count(MixerKind::Mla) {
        write_i32(&status(sp, i, n), &vec![-1; n])?;
    }
    Ok(())
}
pub(super) fn validate(seqs: &[&Glm5NextSession<'_>]) -> Result<()> {
    let c = &seqs[0].weights.config;
    let sp = seqs[0].packed.as_ref().unwrap().sparse.as_ref().unwrap();
    for i in 0..c.block_count(MixerKind::Mla) {
        for (row, value) in read_i32(&status(sp, i, seqs.len()))?
            .into_iter()
            .enumerate()
        {
            if value != crate::metal::SELECT_STATUS_OK {
                return Err(Glm5NextMetalError::KernelValidation {
                    stage: "independent sparse selection",
                    block: mla_block(c, i),
                    row: Some(row),
                    status: value,
                });
            }
        }
    }
    Ok(())
}
pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    seqs: &[&Glm5NextSession<'_>],
    mla: usize,
    stride: usize,
) -> Result<()> {
    let c = &seqs[0].weights.config;
    let sp = seqs[0].packed.as_ref().unwrap().sparse.as_ref().unwrap();
    let n = seqs.len();
    let top = c.selected_pool_count() as usize;
    let slots = c.selection_width() as usize;
    let pools = seqs
        .iter()
        .map(|s| (s.position + 1) / c.indexer_pool as usize)
        .max()
        .unwrap();
    let vector = |t: &MetalTensor| t.view_subrange(0, vec![n as u64]);
    crate::metal::encode_glm_select_top_k_members(
        ctx,
        enc,
        &crate::metal::TopKSelection {
            scores: &sp.scores.view_subrange(0, vec![stride as u64, n as u64]),
            visible_counts: &vector(&sp.visible_pools),
            ids: &sp.pool_ids.view_subrange(0, vec![top as u64, n as u64]),
            counts: &vector(&sp.pool_counts),
            status: &status(sp, mla, n),
        },
        stride,
        pools,
        n,
    )?;
    crate::metal::encode_indexer_expand_selection(
        ctx,
        enc,
        &crate::metal::IndexerSelection {
            pool_ids: &sp.pool_ids.view_subrange(0, vec![top as u64, n as u64]),
            pool_counts: &vector(&sp.pool_counts),
            visible_rows: &vector(&sp.visible_rows),
            row_ids: &sp.row_ids.view_subrange(0, vec![slots as u64, n as u64]),
            row_counts: &vector(&sp.row_counts),
        },
        top,
        slots,
        n,
    )?;
    Ok(())
}

pub(super) fn requested() -> bool {
    enabled()
}
