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
crate::env_flag!(
    /// Batch the independent guarded top-k selection across members. Requires the fused shared/router projections and the SIMD guarded kernel; otherwise members keep their own top-k. 0 restores per-member top-k.
    default_on batch_topk_enabled, "QWEN_FN_MOE_BATCH_TOPK"
);
fn batch_topk_selected() -> bool {
    #[cfg(feature = "perf-ablations")]
    {
        return crate::metal::fn_batch_probe::axis("moe_batch_topk", batch_topk_enabled());
    }
    #[cfg(not(feature = "perf-ablations"))]
    {
        batch_topk_enabled()
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    hidden: u32,
    top_k: u32,
    rows: u32,
    pad: u32,
}
pub(crate) fn encode_into<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: Qwen4ExpMoeMetalWeights<'_>,
    mut members: Vec<&'a mut Qwen4ExpMoeMetalWorkspace>,
    scratch: Option<&Qwen4ExpMoePackedMotorScratch>,
    destinations: &[&MetalTensor],
    direct: bool,
) -> Result<(), Qwen4ExpMoeError> {
    if destinations.len() != members.len() {
        return invalid("native destination/member count differs");
    }
    if direct {
        for (w, destination) in members.iter().zip(destinations) {
            validate_output_destination(ctx, w, destination)?;
        }
    }
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
        for ((w, x), destination) in members.into_iter().zip(inputs).zip(destinations) {
            encode_qwen4exp_moe(ctx, enc, x, weights, w)?
                .output()
                .encode_copy_to(ctx, enc, destination)?;
        }
        return Ok(());
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
    let batch_topk = batch_topk_selected();
    let mut fused_flags = Vec::with_capacity(rows);
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
        fused_flags.push(fused);
        if !(batch_topk && fused) {
            encode_singleton_router(ctx, enc, input, weights, b, fused)?;
        }
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
    if batch_topk {
        if fused_flags.iter().all(|&fused| fused) {
            let logits: Vec<_> = members.iter().map(|w| &w.router_logits).collect();
            let ids: Vec<_> = members.iter().map(|w| &w.topk_ids).collect();
            let topk_weights: Vec<_> = members.iter().map(|w| &w.topk_weights).collect();
            let status: Vec<_> = members.iter().map(|w| &w.route_status).collect();
            if !guarded_topk::encode_members_with_status(
                ctx,
                enc,
                &logits,
                &ids,
                &topk_weights,
                &status,
            )? {
                for w in &members {
                    super::encode_singleton_topk(ctx, enc, g, w)?;
                }
            }
        } else {
            for (w, &fused) in members.iter().zip(&fused_flags) {
                if fused {
                    super::encode_singleton_topk(ctx, enc, g, w)?;
                }
            }
        }
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
            (
                10 + j,
                if direct {
                    destinations[j.min(rows - 1)]
                } else {
                    &w.output
                },
            ),
        ] {
            enc.set_tensor(slot, t);
        }
    }
    enc.dispatch(grid, threads);
    if direct {
        #[cfg(feature = "perf-ablations")]
        crate::metal::fn_batch_probe::observe("direct_outputs", rows);
    } else {
        for (workspace, destination) in members.into_iter().zip(destinations) {
            Qwen4ExpMoeMetalRead { workspace }
                .output()
                .encode_copy_to(ctx, enc, destination)?;
        }
    }
    Ok(())
}
