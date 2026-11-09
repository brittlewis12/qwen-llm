//! Independent sequence axes for the incumbent QSA attention reductions.
use super::*;
crate::env_flag!(
    /// Span independent decode members in QSA logits, softmax, split value and merge launches. 0 keeps private launches. Preserves F32 query/F16 cache and incumbent reduction orders.
    default_on enabled,"QWEN_FN_BATCH_QSA_ATTENTION"
);
fn ids(g: QwenSparseAttentionMetalGeometry, n: usize) -> usize {
    (n / g.ratio).min(g.block_budget()) * g.ratio + n % g.ratio
}
pub(super) fn eligible(
    g: QwenSparseAttentionMetalGeometry,
    members: &[&mut QwenSparseAttentionMetalWorkspace],
    controls: &[(usize, usize)],
) -> bool {
    enabled()
        && decode_reuse::batch_enabled()
        && selected_value::geometry_qualified(g)
        && members.iter().zip(controls).all(|(w, &(_, n))| {
            w.split_decode_scratch.is_some() && split_decode::eligible(g, ids(g, n))
        })
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Split {
    ids: u32,
    capacity: u32,
    splits: u32,
    keys_per_split: u32,
    row_stride: u32,
}
pub(in crate::qwen4exp_qsa) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    members: &[&mut QwenSparseAttentionMetalWorkspace],
    controls: &[(usize, usize)],
) -> Result<(), Qwen4ExpQsaError> {
    let b = members.len();
    let g = members[0].geometry;
    let counts: Vec<_> = controls.iter().map(|&(_, n)| ids(g, n)).collect();
    let logits: Vec<_> = counts.iter().map(|&n| attention_args(g, n)).collect();
    let splits: Vec<_> = counts
        .iter()
        .map(|&n| {
            let s = split_decode::splits_for(n);
            Split {
                ids: n as u32,
                capacity: g.capacity as u32,
                splits: s as u32,
                keys_per_split: n.div_ceil(s).max(1) as u32,
                row_stride: g.output_width() as u32,
            }
        })
        .collect();
    let partials: Vec<_> = members
        .iter()
        .map(|w| {
            w.split_decode_scratch
                .as_ref()
                .expect("eligible QSA scratch")
        })
        .collect();
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_logits")?);
    enc.set_bytes_slice(0, &logits);
    for i in 0..4 {
        let w = &members[i.min(b - 1)];
        for (slot, t) in [
            (1 + i, &w.query),
            (5 + i, &w.key_cache),
            (9 + i, &w.token_ids),
            (13 + i, &w.attention_logits),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(
        MTLSize {
            width: 3 * counts.iter().max().unwrap().div_ceil(32),
            height: 2,
            depth: b,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_softmax")?);
    enc.set_bytes_slice(0, &splits);
    for i in 0..4 {
        let j = i.min(b - 1);
        enc.set_tensor(1 + i, &members[j].attention_logits);
        enc.set_tensor(5 + i, partials[j]);
    }
    enc.set_threadgroup_memory(0, ATTENTION_SCRATCH_BYTES);
    enc.dispatch(
        MTLSize {
            width: 24,
            height: b,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_value")?);
    enc.set_bytes_slice(0, &splits);
    for i in 0..4 {
        let j = i.min(b - 1);
        let w = &members[j];
        for (slot, t) in [
            (1 + i, &w.attention_logits),
            (5 + i, &w.value_cache),
            (9 + i, &w.token_ids),
            (13 + i, partials[j]),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.set_threadgroup_memory(0, 0);
    enc.dispatch(
        MTLSize {
            width: splits.iter().map(|s| s.splits as usize).max().unwrap(),
            height: 2,
            depth: b,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_merge")?);
    enc.set_bytes_slice(0, &splits);
    for i in 0..4 {
        let j = i.min(b - 1);
        let w = &members[j];
        for (slot, t) in [
            (1 + i, partials[j]),
            (5 + i, &w.raw_gate),
            (9 + i, &w.attention),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(
        MTLSize {
            width: 24,
            height: b,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
