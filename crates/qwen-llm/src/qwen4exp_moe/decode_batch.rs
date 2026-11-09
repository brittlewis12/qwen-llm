//! Independent token grid, without the packed-prefill bucket/MMA policy.
use super::*;
use objc2_metal::MTLSize;
crate::env_flag!(
    /// Independent Flash-Next decode native expert token grid. 0 keeps per-member experts. Only B=2..4 with admitted packed transport scratch; never changes scalar or prefill paths.
    default_on enabled, "QWEN_FN_BATCH_EXPERT_GRID"
);
crate::env_flag!(
    /// Four output channels per SIMD for independent Flash-Next native expert grids. 0 keeps the generic token grids.
    default_on tiles_enabled,"QWEN_FN_BATCH_EXPERT_TILES"
);
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    hidden: u32,
    top_k: u32,
    rows: u32,
    pad: u32,
}
pub(crate) fn encode<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: Qwen4ExpMoeMetalWeights<'_>,
    mut members: Vec<&'a mut Qwen4ExpMoeMetalWorkspace>,
    scratch: Option<&Qwen4ExpMoePackedMotorScratch>,
) -> Result<Vec<Qwen4ExpMoeMetalRead<'a>>, Qwen4ExpMoeError> {
    let g = weights.geometry;
    let rows = members.len();
    let Some(scratch) = scratch.filter(|s| {
        enabled()
            && (2..=4).contains(&rows)
            && s.capacity >= rows
            && s.geometry == g
            && crate::metal::all_slots_gate_up_native_supported(
                weights.routed_gate.dtype,
                weights.routed_up.dtype,
            )
            && crate::metal::all_slots_down_native_supported(weights.routed_down.dtype)
    }) else {
        return members
            .into_iter()
            .zip(inputs)
            .map(|(w, x)| encode_qwen4exp_moe(ctx, enc, x, weights, w))
            .collect();
    };
    validate_encoder(ctx, enc)?;
    for (w, x) in members.iter().zip(inputs) {
        if w.state_poisoned || w.geometry != g {
            return invalid("independent MoE member geometry/state differs");
        }
        w.require_idle()?;
        validate_contract(ctx, x, weights, w)?;
        preflight(ctx, weights)?;
    }
    let gather = ctx.pipeline("kernel_fn_batch_moe_gather")?;
    let scatter = ctx.pipeline("kernel_fn_batch_moe_scatter")?;
    for p in [&gather, &scatter] {
        if p.threadExecutionWidth() != 32 || p.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("independent MoE transport needs width32/256 threads");
        }
    }
    let v = scratch.views(rows)?;
    // Reuse the packed shared-output transport as inputs; its ordinary shared
    // expert values remain in each member's separate workspace.
    let x = scratch
        .shared_output
        .view_subrange(0, vec![g.hidden_size as u64, rows as u64]);
    for (w, input) in members.iter_mut().zip(inputs) {
        reserve_command(w, enc)?;
        let b = Qwen4ExpMoeSingletonBuffers {
            guarded_topk: w.guarded_topk,
            router_logits: &w.router_logits,
            topk_ids: &w.topk_ids,
            topk_weights: &w.topk_weights,
            shared_scale: &w.shared_gate,
            route_status: &w.route_status,
            routed_inner: &w.routed_inner,
            routed_expert_output: &w.routed_expert_output,
            shared_inner: &w.shared_inner,
            shared_output: &w.shared_output,
            output: &w.output,
        };
        let fused = projections::try_encode(ctx, enc, input, weights, b)?;
        encode_singleton_router(ctx, enc, input, weights, b, fused)?;
        if fused {
            let n = g.shared_intermediate_size as u64;
            encode_silu_mul_f32(
                ctx,
                enc,
                &w.shared_output.view_subrange(0, vec![n]),
                &w.shared_output.view_subrange(n, vec![n]),
                &w.shared_inner,
            )?;
        } else {
            encode_singleton_shared_gate_up(ctx, enc, input, weights, b)?;
        }
        encode_singleton_shared_down(ctx, enc, weights, b)?;
    }
    let args = Args {
        hidden: g.hidden_size as u32,
        top_k: g.experts_per_token as u32,
        rows: rows as u32,
        pad: 0,
    };
    enc.set_pipeline(&gather);
    enc.set_bytes(0, &args);
    for j in 0..4 {
        let i = j.min(rows - 1);
        let w = &members[i];
        for (slot, t) in [
            (1 + j, inputs[i]),
            (5 + j, &w.topk_ids),
            (9 + j, &w.topk_weights),
            (13 + j, &w.route_status),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    for (i, t) in [&x, &v.topk_ids, &v.topk_weights, &v.route_status]
        .into_iter()
        .enumerate()
    {
        enc.set_tensor(17 + i, t);
    }
    let grid = MTLSize {
        width: g.hidden_size.div_ceil(256),
        height: rows,
        depth: 1,
    };
    let threads = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    enc.dispatch(grid, threads);
    let inner = v.routed_inner.view_subrange(
        0,
        vec![
            g.routed_intermediate_size as u64,
            (g.experts_per_token * rows) as u64,
        ],
    );
    let expert_output = v.routed_expert_output.view_subrange(
        0,
        vec![g.hidden_size as u64, (g.experts_per_token * rows) as u64],
    );
    let gate_up = if tiles_enabled() {
        crate::metal::encode_fn_batch_gate_up
    } else {
        crate::metal::encode_all_slots_gate_up_swiglu_native_fused_tokens
    };
    let down = if tiles_enabled() {
        crate::metal::encode_fn_batch_down
    } else {
        crate::metal::encode_all_slots_down_native_tokens
    };
    gate_up(
        ctx,
        enc,
        weights.routed_gate,
        weights.routed_up,
        &x,
        &v.topk_ids,
        &v.route_status,
        &inner,
        (
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
        ),
        rows,
        f32::MAX,
    )?;
    down(
        ctx,
        enc,
        weights.routed_down,
        &inner,
        &v.topk_ids,
        &v.route_status,
        &expert_output,
        (
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            g.experts_per_token,
        ),
        rows,
    )?;
    encode_moe_weighted_sum_packed_f32(
        ctx,
        enc,
        &expert_output,
        &v.topk_weights,
        &v.output,
        g.hidden_size,
        g.experts_per_token,
        rows,
    )?;
    enc.set_pipeline(&scatter);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, &v.output);
    for j in 0..4 {
        let w = &members[j.min(rows - 1)];
        for (slot, t) in [
            (2 + j, &w.shared_output),
            (6 + j, &w.shared_gate),
            (10 + j, &w.output),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(grid, threads);
    Ok(members
        .into_iter()
        .map(|workspace| Qwen4ExpMoeMetalRead { workspace })
        .collect())
}
