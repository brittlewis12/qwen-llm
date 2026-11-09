use super::*;
use crate::qwen4exp_metal::decode_projection::Scratch;
pub(crate) fn encode<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: GatedDeltaNetMetalWeights<'_>,
    mut members: Vec<&'a mut GatedDeltaNetMetalWorkspace>,
    scratch: &Scratch,
) -> Result<Vec<GatedDeltaNetMetalRead<'a>>, Qwen4ExpGdnError> {
    for (w, x) in members.iter().zip(inputs) {
        validate_decode(ctx, enc, x, weights, w)?;
    }
    for w in &mut members {
        reserve_command(w, enc)?;
    }
    let qkv: Vec<_> = members.iter().map(|w| &w.qkv).collect();
    scratch.project(ctx, enc, weights.qkv, inputs, &qkv)?;
    let gates: Vec<_> = members.iter().map(|w| &w.gate).collect();
    scratch.project(ctx, enc, weights.gate, inputs, &gates)?;
    if causal_enabled() && weights.geometry.head_dim == 128 {
        causal(ctx, enc, inputs, weights, &members)?;
    } else {
        for (w, x) in members.iter().zip(inputs) {
            encode_recurrence(ctx, enc, x, weights, w)?;
        }
    }
    let normalized: Vec<_> = members.iter().map(|w| &w.normalized).collect();
    let outputs: Vec<_> = members.iter().map(|w| &w.output).collect();
    scratch.project(ctx, enc, weights.output, &normalized, &outputs)?;
    Ok(members
        .into_iter()
        .map(|workspace| GatedDeltaNetMetalRead { workspace })
        .collect())
}

crate::env_flag!(
    /// Independent GDN recurrence and gated RMS normalization grids over every member. 0 restores per-member dispatches; scalar and prefill paths are unchanged.
    default_on causal_enabled,"QWEN_FN_BATCH_GDN"
);
fn causal(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: GatedDeltaNetMetalWeights<'_>,
    members: &[&mut GatedDeltaNetMetalWorkspace],
) -> Result<(), Qwen4ExpGdnError> {
    let g = weights.geometry;
    let b = members.len();
    for (w, x) in members.iter().zip(inputs) {
        encode_recurrence_prepare(ctx, enc, x, weights, w)?;
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        value: u32,
        key: u32,
    }
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_gdn_decay")?);
    enc.set_bytes(
        0,
        &Args {
            value: g.value_heads as u32,
            key: g.key_heads as u32,
        },
    );
    let values: Vec<_> = members
        .iter()
        .map(|w| {
            w.qkv_conv
                .view_subrange((2 * g.key_width()) as u64, vec![g.value_width() as u64])
        })
        .collect();
    for i in 0..4 {
        let j = i.min(b - 1);
        let w = &members[j];
        for (slot, t) in [
            (1 + i, &w.query_norm),
            (5 + i, &w.key_norm),
            (9 + i, &values[j]),
            (13 + i, &w.decay),
            (17 + i, &w.beta),
            (21 + i, &w.delta_state),
            (25 + i, &w.recurrent),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(
        MTLSize {
            width: 128,
            height: g.value_heads,
            depth: b,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Norm {
        heads: u32,
        eps: f32,
    }
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_gdn_rms_gate")?);
    enc.set_bytes(
        0,
        &Norm {
            heads: g.value_heads as u32,
            eps: g.eps * g.head_dim as f32,
        },
    );
    enc.set_tensor(1, weights.norm);
    for i in 0..4 {
        let w = &members[i.min(b - 1)];
        for (slot, t) in [
            (2 + i, &w.recurrent),
            (6 + i, &w.gate),
            (10 + i, &w.normalized),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(
        MTLSize {
            width: g.value_heads.div_ceil(4),
            height: b,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    Ok(())
}
