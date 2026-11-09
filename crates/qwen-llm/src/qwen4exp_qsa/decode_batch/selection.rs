use super::*;
crate::env_flag!(
    /// Independent QSA index-score, exact compact selector and ID expansion grids. 0 retains per-member launches. Deep selected contexts and incumbent compact/parallel rollbacks only.
    default_on enabled,"QWEN_FN_BATCH_QSA_SELECTION"
);
pub(super) fn eligible(
    ctx: &MetalContext,
    g: QwenSparseAttentionMetalGeometry,
    controls: &[(usize, usize)],
) -> bool {
    if !enabled()
        || !expand_ids_parallel()
        || g.block_budget() > EXPAND_PAR_MAX_BLOCKS
        || controls
            .iter()
            .any(|&(_, n)| n / g.ratio <= g.block_budget())
    {
        return false;
    }
    let threads = index_reuse::selector_threads(ctx, g.block_capacity(), 1);
    prefill_reuse::selector_pipeline(ctx, g.block_capacity(), g.block_budget(), 1, threads, true)
        .is_ok_and(|p| p.is_some())
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Score {
    visible: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Select {
    capacity: u32,
    top_k: u32,
    queries: u32,
    ranked: u32,
}
pub(in crate::qwen4exp_qsa) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    members: &[&mut QwenSparseAttentionMetalWorkspace],
    controls: &[(usize, usize)],
) -> Result<(), Qwen4ExpQsaError> {
    let b = members.len();
    let g = members[0].geometry;
    let scores: Vec<_> = controls
        .iter()
        .map(|&(_, n)| Score {
            visible: (n / g.ratio) as u32,
        })
        .collect();
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_index")?);
    enc.set_bytes_slice(0, &scores);
    for i in 0..4 {
        let w = &members[i.min(b - 1)];
        for (slot, t) in [
            (1 + i, &w.index_query),
            (5 + i, &w.compressed_index_keys),
            (9 + i, &w.scores),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(
        MTLSize {
            width: scores
                .iter()
                .map(|s| s.visible as usize)
                .max()
                .unwrap()
                .div_ceil(LOGITS_SIMDGROUPS_PER_TG),
            height: b,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    let p = ctx.pipeline("kernel_fn_batch_qsa_select")?;
    let threads = index_reuse::selector_threads(ctx, g.block_capacity(), 1);
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &Select {
            capacity: g.block_capacity() as u32,
            top_k: g.block_budget() as u32,
            queries: 1,
            ranked: 0,
        },
    );
    for i in 0..4 {
        let w = &members[i.min(b - 1)];
        for (slot, t) in [
            (1 + i, &w.scores),
            (5 + i, &w.visible_blocks),
            (9 + i, &w.selected_blocks),
            (13 + i, &w.selected_count),
            (17 + i, &w.selector_status),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.set_threadgroup_memory(0, threads * 4);
    enc.set_threadgroup_memory(1, threads * 4);
    enc.dispatch(
        MTLSize {
            width: b,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    let args: Vec<_> = controls
        .iter()
        .map(|&(_, n)| id_args(g, n / g.ratio, n))
        .collect();
    let p = ctx.pipeline("kernel_fn_batch_qsa_expand")?;
    enc.set_pipeline(&p);
    enc.set_bytes_slice(0, &args);
    for i in 0..4 {
        let w = &members[i.min(b - 1)];
        enc.set_tensor(1 + i, &w.selected_blocks);
        enc.set_tensor(5 + i, &w.token_ids);
    }
    enc.set_threadgroup_memory(0, 0);
    enc.set_threadgroup_memory(1, 0);
    enc.dispatch(
        MTLSize {
            width: b,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: p.maxTotalThreadsPerThreadgroup().min(1024),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
